"""A multimodel of Whisper, Qwen3.5 (with tools) and Kokoro builds and starts as one model, and a
member whose names another already has is refused."""
import pytest
from test_startup_graph_validation import (
    _config,
    _drop_stubbed_imports,  # noqa: F401  (a fixture the factories' stubs need)
    _kokoro,
    _qwen3_5,
    _start_conductor,
    _whisper,
)

from mstar.engine.resources.kv.config import prefix_caches_by_rank
from mstar.model.base import names_checkpoint
from mstar.model.components.text_glue import GLUE_NODE, STREAM_WALK, TRANSCRIPT_STREAM
from mstar.model.keys import SPEAK_TEXT
from mstar.model.kokoro.config import FRONT_NODE, FRONT_WALK
from mstar.model.multimodel import Link, MultiModel
from mstar.model.whisper.config import (
    DECODE_WALK as ASR_DECODE,
)
from mstar.model.whisper.config import (
    DECODER_NODE,
    DETECT_LANGUAGE_WALK,
    NEW_TOKEN,
    PREFILL_PROMPT_WALK,
    PREFILL_WALK,
)
from mstar.streaming.topology import StreamingGraphEdge

SPEECH = Link(
    "asr", DECODER_NODE, NEW_TOKEN, (PREFILL_WALK, DETECT_LANGUAGE_WALK, PREFILL_PROMPT_WALK, ASR_DECODE),
    "llm", STREAM_WALK,
)


def _voice_agent(tmp_path, monkeypatch, links=()):
    asr = _whisper("whisper", {}, tmp_path, monkeypatch)
    llm = _qwen3_5("qwen", {"tools": ["python"]}, tmp_path, monkeypatch)
    if links:
        asr.stream_out, llm.stream_in = True, "transcript"
    return MultiModel({"asr": asr, "llm": llm, "tts": _kokoro("kokoro", {}, tmp_path, monkeypatch)}, links=links)


def test_whisper_qwen_and_kokoro_start_as_one_model(tmp_path, monkeypatch, _drop_stubbed_imports):  # noqa: F811
    model = _voice_agent(tmp_path, monkeypatch)
    groups = [
        {"node_names": ["audio_encoder", "decoder"], "ranks": [0]},
        {"node_names": ["LLM", "vision_encoder"], "ranks": [1]},
        {"node_names": ["Router", "Tool"], "ranks": [2]},
        {"node_names": ["kokoro"], "ranks": [3]},
    ]
    conductor = _start_conductor(model, _config(tmp_path, groups), tmp_path, monkeypatch)
    assert conductor.worker_graphs
    assert sorted(p.name for p in model.get_partitions()) == ["asr.default", "llm.default", "tts.default"]
    # each member's plan runs in its own partition, in its own walks
    plan = model.walk_plan("llm.default", ["text"], ["text"], {"text_inputs": [object()]}, {})
    assert [step.walk for step in plan] == ["prefill_text", "decode", "route", "tool", "prefill_text"]
    assert {"asr_decode", "decode", "synth"} <= set(model.get_graph_walk_graphs())
    assert {"asr_kv_cache", "kv_cache"} <= {spec.resource_key for spec in model.get_node_resources()}
    # Qwen's prefix cache opens: the deployment names the weights it hangs from (a shut cache serves no multi-turn
    # reuse and counts no lookups)
    kv = [spec for spec in model.get_node_resources() if spec.resource_key == "kv_cache"]
    assert prefix_caches_by_rank(kv, [([1], 1, ["LLM"])], names_checkpoint(model))[1] == {"kv_cache"}


def test_a_member_that_repeats_another_members_names_is_refused(tmp_path, monkeypatch, _drop_stubbed_imports):  # noqa: F811
    first = _qwen3_5("qwen", {}, tmp_path, monkeypatch)
    second = _qwen3_5("qwen", {}, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"members share walk names: .*'decode' \(a, b\)"):
        MultiModel({"a": first, "b": second})


def test_a_spoken_prompt_runs_whisper_into_a_glued_streamed_prefill_and_a_typed_one_is_qwens_alone(
    tmp_path, monkeypatch, _drop_stubbed_imports,  # noqa: F811
):
    model = _voice_agent(tmp_path, monkeypatch, links=[SPEECH])
    groups = [
        {"node_names": ["audio_encoder", "decoder"], "ranks": [0]},
        # (a walk a stream drives is served by one worker graph: the glue shares the LLM's rank)
        {"node_names": ["LLM", "vision_encoder", GLUE_NODE], "ranks": [1]},
        {"node_names": ["Router", "Tool"], "ranks": [2]},
        {"node_names": ["kokoro"], "ranks": [3]},
    ]
    assert _start_conductor(model, _config(tmp_path, groups), tmp_path, monkeypatch).worker_graphs

    walks = model.get_graph_walk_graphs()
    # Whisper's tokens stream from its prompt walks and its decode loop; the glue is first in the streamed walk
    for walk in SPEECH.walks:
        assert any(isinstance(e, StreamingGraphEdge) and e.name == NEW_TOKEN
                   for n in walks[walk].get_nodes().values() for e in n.outputs)
    assert list(walks[STREAM_WALK].get_nodes()) == [GLUE_NODE, "LLM"]
    (link,) = [c for c in model.get_partition_topology().connections if c.edge_name == NEW_TOKEN]
    assert (link.from_partition, link.to_partition) == ("asr.default", "llm.default")

    speech, text = ["audio"], ["text"]
    signals = {"text_inputs": [object()]}
    assert model.walk_plan("asr.default", speech, text, signals, {})
    assert model.walk_plan("asr.default", text, text, signals, {}) == []
    spoken = model.walk_plan("llm.default", speech, text, signals, {TRANSCRIPT_STREAM: True})
    assert [s.walk for s in spoken][:4] == ["prefill_text", STREAM_WALK, "prefill_text", "decode"]
    assert model.max_input_files()["audio"] == 1
    assert model.postprocessor(NEW_TOKEN) is model.members["asr"]
    assert model.postprocessor("new_token") is model.members["llm"]
    assert model.postprocessor("audio_chunk") is model.members["tts"]


SPEAK = Link("llm", "Router", SPEAK_TEXT, ("route",), "tts", FRONT_WALK, glue=False)


def test_a_reply_that_speaks_streams_text_to_kokoros_front_and_its_audio_to_the_client(
    tmp_path, monkeypatch, _drop_stubbed_imports,  # noqa: F811
):
    asr = _whisper("whisper", {}, tmp_path, monkeypatch)
    llm = _qwen3_5("qwen", {"tools": ["python", "speak"]}, tmp_path, monkeypatch)
    tts = _kokoro("kokoro", {}, tmp_path, monkeypatch)
    asr.stream_out, llm.stream_in, tts.stream_in = True, "transcript", True
    model = MultiModel({"asr": asr, "llm": llm, "tts": tts}, links=[SPEECH, SPEAK])
    groups = [
        {"node_names": ["audio_encoder", "decoder"], "ranks": [0]},
        {"node_names": ["LLM", "vision_encoder", GLUE_NODE], "ranks": [1]},
        {"node_names": ["Router", "Tool", FRONT_NODE], "ranks": [2]},
        {"node_names": ["kokoro"], "ranks": [3]},
    ]
    assert _start_conductor(model, _config(tmp_path, groups), tmp_path, monkeypatch).worker_graphs

    walks = model.get_graph_walk_graphs()
    assert any(isinstance(e, StreamingGraphEdge) and e.name == SPEAK_TEXT and e.next_node == FRONT_NODE
               for n in walks["route"].get_nodes().values() for e in n.outputs)
    connections = {(c.from_partition, c.to_partition, c.edge_name) for c in model.get_partition_topology().connections}
    assert ("llm.default", "tts.front", SPEAK_TEXT) in connections
    assert ("tts.front", "tts.default", "phoneme_ids") in connections

    signals = {"text_inputs": [object()]}
    speaking, silent = ["text", "audio"], ["text"]
    assert [s.walk for s in model.walk_plan("tts.front", ["text"], speaking, signals, {})] == [FRONT_WALK]
    assert [s.walk for s in model.walk_plan("tts.default", ["text"], speaking, signals, {})] == ["synth_stream"]
    assert model.walk_plan("tts.front", ["text"], silent, signals, {}) == []
    plan = model.walk_plan("llm.default", ["text"], speaking, signals, {})
    assert next(s for s in plan if s.walk == "route").step_metadata["speak"] is True
    assert next(s for s in model.walk_plan("llm.default", ["text"], silent, signals, {}) if s.walk == "route"
                ).step_metadata["speak"] is False
    assert model.postprocessor("audio_chunk") is tts


def test_a_member_that_can_speak_needs_a_link_for_its_text(tmp_path, monkeypatch, _drop_stubbed_imports):  # noqa: F811
    llm = _qwen3_5("qwen", {"tools": ["python", "speak"]}, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="can speak"):
        MultiModel({"llm": llm, "tts": _kokoro("kokoro", {}, tmp_path, monkeypatch)})



def test_a_model_that_starts_agents_starts_as_one_deployment_with_no_node_of_its_own_for_them(
    tmp_path, monkeypatch, _drop_stubbed_imports,  # noqa: F811
):
    model = _qwen3_5("qwen", {"tools": ["python", "spawn_agent"]}, tmp_path, monkeypatch)
    groups = [{"node_names": ["LLM", "vision_encoder"], "ranks": [0]}, {"node_names": ["Router", "Tool"], "ranks": [1]}]
    conductor = _start_conductor(model, _config(tmp_path, groups), tmp_path, monkeypatch)
    assert conductor.worker_graphs
    plan = model.walk_plan("LLM", ["text"], ["text"], {"text_inputs": [object()]}, {})
    assert any(step.spawns for step in plan) and any(step.joins for step in plan)


def _ingested(model, tmp_path, monkeypatch, input_modalities, signals, kwargs):
    """The real conductor ingests one request of the multimodel; returns the conductor (a request the
    model's layouts do not cover fails with a 500 in `refuse_undeclared_labels`)."""
    from test_ingest_unpersist import ConductorMessageType, NewRequestConductor, _conductor, _info, _msg, _run

    walks = model.get_graph_walk_graphs()
    groups = [sorted({n for w in p.graph_walks for n in walks[w].get_nodes()}) for p in model.get_partitions()]
    c = _conductor(model, [{"node_names": g, "ranks": [i]} for i, g in enumerate(groups)], tmp_path, monkeypatch)
    _run(c, [[_msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
        request_id="r", initial_signals={name: [_info(i + 1)] for i, name in enumerate(signals)},
        initial_input_modalities=list(input_modalities), initial_output_modalities=["text", "audio"],
        input_metadata={}, model_kwargs=kwargs,
    ))]])
    kinds = [getattr(msg.message_type, "name", msg.message_type) for _, msg in c.comm.sent]
    failed = [msg.body for _, msg in c.comm.sent if getattr(msg.message_type, "name", "") == "request_failed"]
    assert "NEW_REQUEST" in kinds and not failed, failed
    return c


@pytest.mark.parametrize("request_kind", ["typed", "spoken", "child"])
def test_a_request_opens_only_the_resources_of_the_members_it_uses(
    tmp_path, monkeypatch, _drop_stubbed_imports, request_kind,  # noqa: F811
):
    """A typed request, an agent's child, never opens Whisper's cross-attention cache or Kokoro; a spoken one
    opens Whisper's too. (The conductor refuses a request whose layouts miss a label it opens.)"""
    from mstar.conductor.request_info import AGENT_CHILD
    from mstar.model.base import LabelLayout, Span
    from mstar.model.whisper import config as asr

    asr_model = _whisper("whisper", {}, tmp_path, monkeypatch)
    llm = _qwen3_5("qwen", {"tools": ["python", "speak", "spawn_agent"]}, tmp_path, monkeypatch)
    asr_model.stream_out, llm.stream_in = True, "transcript"
    tts = _kokoro("kokoro", {}, tmp_path, monkeypatch)
    tts.stream_in = True
    model = MultiModel({"asr": asr_model, "llm": llm, "tts": tts}, links=[SPEECH, SPEAK])
    layout = llm.kv_layout({}, 5)
    if request_kind == "spoken":
        layout = {**layout, asr.CROSS_KV_CACHE: {asr.CONTEXT_LABEL: LabelLayout(
            spans=(Span("rows", asr.PREFILL_WALK, 8, 8, "encoder_states"),))}, asr.KV_CACHE: {"main": LabelLayout(
                spans=(Span("ids", asr.PREFILL_WALK, 3, 3, "asr_text"),), growth=((asr.DECODE_LOOP, 1),))}}
    layout = {key: {label: each.row() for label, each in by_label.items()} for key, by_label in layout.items()}
    kwargs = {"kv_layout": layout, **({TRANSCRIPT_STREAM: True} if request_kind == "spoken" else {})}
    signals = ("audio", "asr_text", "text_inputs", "suffix") if request_kind == "spoken" else ("text_inputs",)
    c = _ingested(model, tmp_path, monkeypatch, ["audio"] if request_kind == "spoken" else ["text"], signals, kwargs)
    if request_kind == "child":
        # the parent's router step persists a prompt, and the conductor starts the child from it
        from test_ingest_unpersist import _info

        from mstar.conductor.request_info import WalkStep

        parent = c.requests["r"]
        parent.persist_signals["agent_0"] = [_info(900)]
        c._start_children("r", WalkStep("route", spawns=("agent_0",)))
        assert "r-agent-0" in c.requests and not parent.child_errors, parent.child_errors
        assert c.requests["r-agent-0"].model_kwargs[AGENT_CHILD] == "r"


def test_a_link_whose_stream_the_target_does_not_wait_for_is_refused(tmp_path, monkeypatch, _drop_stubbed_imports):  # noqa: F811
    asr = _whisper("whisper", {}, tmp_path, monkeypatch)
    llm = _qwen3_5("qwen", {"tools": ["python"]}, tmp_path, monkeypatch)
    asr.stream_out, llm.stream_in = True, "voice"
    with pytest.raises(ValueError, match="would take the typed path"):
        MultiModel({"asr": asr, "llm": llm, "tts": _kokoro("kokoro", {}, tmp_path, monkeypatch)}, links=[SPEECH])


def test_usage_counts_a_spoken_turns_whisper_prompt_beside_the_replys(tmp_path, monkeypatch, _drop_stubbed_imports):  # noqa: F811
    import torch

    model = _voice_agent(tmp_path, monkeypatch, links=[SPEECH])
    tensors = {"text_inputs": [torch.zeros(3, dtype=torch.long)], "asr_text": [torch.zeros(4, dtype=torch.long)]}
    assert model.prompt_tokens(tensors, {}) == 7

def test_a_model_that_starts_agents_starts_as_one_deployment_with_no_node_of_its_own_for_them(
    tmp_path, monkeypatch, _drop_stubbed_imports,  # noqa: F811
):
    model = _qwen3_5("qwen", {"tools": ["python", "spawn_agent"]}, tmp_path, monkeypatch)
    groups = [{"node_names": ["LLM", "vision_encoder"], "ranks": [0]}, {"node_names": ["Router", "Tool"], "ranks": [1]}]
    conductor = _start_conductor(model, _config(tmp_path, groups), tmp_path, monkeypatch)
    assert conductor.worker_graphs
    plan = model.walk_plan("LLM", ["text"], ["text"], {"text_inputs": [object()]}, {})
    assert any(step.spawns for step in plan) and any(step.joins for step in plan)
