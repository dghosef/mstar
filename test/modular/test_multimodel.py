"""Two models served as one multimodel: their graphs merge, a link streams one
member's output into the other, and each member keeps its own names."""
import pytest
import torch

from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    StreamingConnectionState,
)
from mstar.graph.base import GraphEdge, GraphNode
from mstar.model.base import ForwardPassArgs
from mstar.model.multimodel import Link, MultiModel
from mstar.model.utils import TextOutput
from mstar.testing_utils.model import StubModel


class _Letters:
    def to_text_bytes(self, ids):
        return bytes(96 + i for i in ids)


class _Member(StubModel):
    """One node in one walk, in this model's own partition."""

    def __init__(self, node, walk, partition="default", persist=False):
        self.walk, self.partition, self.seen = walk, partition, []
        # one graph, handed out every call, as a model that builds it once does
        self.graphs = {walk: GraphNode(name=node, input_names=["x"], outputs=[
            GraphEdge(next_node="emit_to_client", name="out", output_modality="text", persist=persist),
        ])}

    def text_output(self):
        return TextOutput(_Letters(), stop_token_ids={9})

    def get_graph_walk_graphs(self):
        return self.graphs

    def get_partitions(self):
        return [PartitionDefinition(self.partition, {self.walk}, initial_walk=self.walk)]

    def get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals,
                                        incoming_connections=None):
        self.seen.append((partition_name, [(c.from_partition, c.to_partition) for c in incoming_connections]))
        return ForwardPassArgs(CurrentForwardConductorMetadata(graph_walk=self.walk, is_prefill=False), [], [])


def test_a_link_streams_one_member_into_the_other():
    asr, llm = _Member("ear", "listen"), _Member("brain", "think", partition="LLM")
    multimodel = MultiModel({"asr": asr, "llm": llm}, links=[Link("ear", "out", "brain", input="x")])

    walks = multimodel.get_graph_walk_graphs()
    assert set(walks) == {"listen", "think"}
    (edge,) = [e for e in walks["listen"].get_nodes()["ear"].outputs if e.is_streaming]
    assert (edge.next_node, edge.name, edge.target_partition) == ("brain", "x", "llm.LLM")
    assert not any(e.is_streaming for e in asr.graphs["listen"].outputs)  # the member's graph is untouched

    partitions = {p.name: p for p in multimodel.get_partitions()}
    assert set(partitions) == {"asr.default", "llm.LLM"}
    assert partitions["llm.LLM"].producer_partitions == ["asr.default"]
    (conn,) = multimodel.get_partition_topology().connections
    assert (conn.from_partition, conn.to_partition, conn.edge_name) == ("asr.default", "llm.LLM", "x")

    # each member's transition is called with its own names; the link's producer keeps the multimodel's
    state = StreamingConnectionState("asr.default", "llm.LLM", "x")
    multimodel.get_partition_forward_pass_args("llm.LLM", None, {}, [state])
    assert llm.seen == [("LLM", [("asr.default", "LLM")])]


def test_an_open_link_defers_its_member_and_reads_the_producers_text():
    asr, llm = _Member("ear", "listen", persist=True), _Member("brain", "think", partition="LLM")
    multimodel = MultiModel({"asr": asr, "llm": llm}, links=[Link("ear", "out", "brain", open=True)])

    assert multimodel.get_partition_topology().connections == []  # no stream between them
    assert not any(e.is_streaming for e in multimodel.get_graph_walk_graphs()["listen"].get_nodes()["ear"].outputs)
    partitions = {p.name: p for p in multimodel.get_partitions()}
    assert (partitions["llm.LLM"].open_after, partitions["llm.LLM"].open_signal) == ("asr.default", "out")
    assert partitions["asr.default"].open_after is None

    # the producer's own detokenizer, without its stop ids
    assert multimodel.open_text("llm.LLM", torch.tensor([8, 9, 5, 9, 3])) == "hec"
    assert multimodel.open_member("llm.LLM") is llm

    with pytest.raises(ValueError, match="not a signal 'asr' persists"):
        MultiModel({"asr": _Member("ear", "listen"), "llm": llm}, links=[Link("ear", "out", "brain", open=True)])


def test_members_that_share_a_name_are_refused():
    with pytest.raises(ValueError, match="share walk names: 'decode'"):
        MultiModel({"a": _Member("a_llm", "decode"), "b": _Member("b_llm", "decode")})


def _whisper_and_qwen(asr_scope="", llm_scope=""):
    """The real classes, built from their configs without weights."""
    from lora_module_helpers import tiny_config

    from mstar.model.keys import Scope
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    from mstar.model.whisper.config import WhisperModelConfig
    from mstar.model.whisper.whisper_model import WhisperModel

    asr = object.__new__(WhisperModel)
    asr.config, asr.scope = WhisperModelConfig(), Scope(asr_scope)
    llm = object.__new__(Qwen3_5DenseModel)
    llm.config, llm.vision_config, llm.scope = tiny_config(), None, Scope(llm_scope)
    return asr, llm


def test_whisper_and_qwen_are_members_only_when_scoped():
    with pytest.raises(ValueError, match="share walk names: 'decode'"):
        MultiModel(dict(zip(("asr", "llm"), _whisper_and_qwen(), strict=True)))

    asr, llm = _whisper_and_qwen("asr.", "llm.")
    multimodel = MultiModel({"asr": asr, "llm": llm})
    walks = multimodel.get_graph_walk_graphs()
    assert {"asr.decode", "llm.decode", "asr.prefill", "llm.prefill_text"} <= set(walks)
    assert walks["llm.decode"].name == "llm.decode_loop" and walks["asr.decode"].name == "asr.decode_loop"
    assert {"asr.sampler", "llm.sampler", "asr.kv_cache", "llm.kv_cache"} <= {
        s.resource_key for s in multimodel.get_node_resources()
    }
    assert set(multimodel.get_loop_caps()) == {"asr.decode_loop", "llm.decode_loop"}

    # a member is given no other member's persisted signal, and reads its own by its scoped name
    meta = CurrentForwardConductorMetadata(graph_walk="llm.prefill_text", is_prefill=True)
    args = multimodel.get_partition_forward_pass_args(
        "llm.default", meta, {"asr.new_token": ["whisper's"], "llm.new_token": ["qwen's"]},
    )
    assert args.full_metadata.graph_walk == "llm.decode" and args.inputs[0].tensor_info == ["qwen's"]
