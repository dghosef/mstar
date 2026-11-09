"""Text that one model streams out and another reads: a speech model's transcript into an LLM's prefill.

``MultiModel`` links a producer's token stream to a consumer's streamed prefill walk and puts the
``Glue`` node first in that walk (`glued_walk`). The node turns the producer's token ids into text,
incrementally, and the text into the consumer's token ids (`GlueSubmodule`). The two models have
different vocabularies, so ids cannot cross as they are.

Each pass takes a chunk of the producer's ids and emits the consumer's ids for the words it can
finish, holding the last word back (a BPE merge does not cross a space, so words tokenize apart;
a split word would not), and a partial UTF-8 character with it. A pass always emits at least one id,
because the walk it is in completes only when the consumer's node has run: a pass with nothing
finished emits a space, which the template ignores. The last pass (the producer's stream ended)
flushes the held word. The consumer's prefill walk after the stream carries the template's suffix.
"""
from __future__ import annotations

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import ExecutingBatch
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Sequential
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    InputMetadata,
    ModelInputsFromEngine,
    NodeSubmodule,
)

GLUE_NODE = "Glue"
# the empty edge the conductor sends with each pass of a stream-fed step, which tells the worker its walk
GLUE_TICK = "stream_tick"
# a request's streamed prefill, the template's suffix it persists, and the marker that its prompt is spoken
STREAM_WALK = "prefill_stream"
SUFFIX = "suffix"
TRANSCRIPT_STREAM = "transcript_stream"
TRANSCRIPT_RESERVE = "transcript_reserve"
# the most ids a transcript puts into the consumer's KV (and the reserve it asks for)
MAX_TRANSCRIPT_TOKENS = 2048


def glued_walk(walk: GraphSection, signal: str, input_name: str = "text_inputs") -> GraphSection:
    """``walk`` (the consumer's streamed prefill, a node reading ``input_name``) with the ``Glue``
    node ahead of it, reading the producer's ``signal``."""
    glue = GraphNode(
        name=GLUE_NODE, input_names=[signal, GLUE_TICK],
        outputs=[GraphEdge(next_node=next(iter(walk.get_nodes())), name=input_name)],
    )
    return Sequential([glue, walk])


class GlueSubmodule(NodeSubmodule):
    """The ``Glue`` node: the producer's ids in, the consumer's ids out."""

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, source_tokenizer, target_tokenizer, device: str | torch.device = "cpu"):
        super().__init__()
        self.source, self.target = source_tokenizer, target_tokenizer
        self._out_device = torch.device(device)
        # the device the node's outputs are made on (`get_device` reads a parameter)
        self._probe = torch.nn.Parameter(torch.zeros(1, device=self._out_device), requires_grad=False)
        self._space = target_tokenizer.encode(" ", add_special_tokens=False)[:1]

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[ARNodeInputs]) -> bool:
        return True

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA, is_final_stream_chunk: bool = False, **kwargs,
    ) -> ARNodeInputs:
        chunk = inputs.get(next((n for n in inputs if n != GLUE_TICK), ""), [])
        ids = torch.cat([t.reshape(-1) for t in chunk]).tolist() if chunk else []
        return ARNodeInputs(input_seq_len=0, kwargs={
            "rid": fwd_info.rid_handle, "ids": ids, "final": is_final_stream_chunk,
        })

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict:
        return {"rows": [row.kwargs for row in inputs]}

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows=(), **kwargs) -> NameToTensorList:
        (rid,) = engine_inputs.request_ids
        return self.forward_batched(graph_walk, engine_inputs, rows=rows)[rid]

    def _next_ids(self, rid, new_ids: list[int], final: bool) -> list[int]:
        """The consumer's ids for what this pass finishes of the transcript."""
        state = self.request_state(rid)
        heard: list[int] = state.kwargs.setdefault("heard", [])
        sent: int = state.kwargs.get("sent", 0)   # characters of the text already turned into ids
        heard.extend(new_ids)
        # the turn's text starts after the template's own newline: not at the transcript's leading space
        text = self.source.decode(heard, skip_special_tokens=True).lstrip()
        # a partial character ends in the replacement character; the last word is held
        if not final:
            text = text.rstrip("�")
            cut = max(text.rfind(" "), text.rfind("\n"))
            text = text[:cut] if cut > sent else text[:sent]
        else:
            text = text.rstrip()
        piece = text[sent:]
        if not piece and not final:
            return self._space
        state.kwargs["sent"] = len(text)
        ids = self.target.encode(piece, add_special_tokens=False) if piece else []
        sent_total = state.kwargs.get("ids_sent", 0) + len(ids or self._space)
        if sent_total > MAX_TRANSCRIPT_TOKENS:
            ids = ids[: max(1, MAX_TRANSCRIPT_TOKENS - state.kwargs.get("ids_sent", 0))]
            sent_total = min(sent_total, MAX_TRANSCRIPT_TOKENS)
        state.kwargs["ids_sent"] = sent_total
        return ids or self._space

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows=(), **kwargs,
    ) -> dict[str, NameToTensorList]:
        out = {}
        for rid, row in zip(engine_inputs.request_ids, rows, strict=True):
            ids = self._next_ids(row["rid"], row["ids"], row["final"])
            out[rid] = {"text_inputs": [torch.tensor(ids, dtype=torch.long).to(self._out_device)]}
        return out
