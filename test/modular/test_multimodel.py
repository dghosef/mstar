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

from mstar.model.multimodel import MultiModel


def _voice_agent(tmp_path, monkeypatch):
    asr = _whisper("whisper", {}, tmp_path, monkeypatch)
    llm = _qwen3_5("qwen", {"tools": ["python"]}, tmp_path, monkeypatch)
    return MultiModel({"asr": asr, "llm": llm, "tts": _kokoro("kokoro", {}, tmp_path, monkeypatch)})


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
    # an output's bytes are made by the member that emits its signal
    assert model.postprocessor("asr_token") is model.members["asr"]
    assert model.postprocessor("new_token") is model.members["llm"]
    assert model.postprocessor("audio_chunk") is model.members["tts"]


def test_a_member_that_repeats_another_members_names_is_refused(tmp_path, monkeypatch, _drop_stubbed_imports):  # noqa: F811
    first = _qwen3_5("qwen", {}, tmp_path, monkeypatch)
    second = _qwen3_5("qwen", {}, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"members share walk names: .*'decode' \(a, b\)"):
        MultiModel({"a": first, "b": second})
