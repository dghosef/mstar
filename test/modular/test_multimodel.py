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

from mstar.model.components.text_glue import GLUE_NODE, STREAM_WALK, TRANSCRIPT_STREAM
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
