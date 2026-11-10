"""The Python tool: Qwen3.5's tool grammar, the continuation text it writes, the sandboxed run,
and the model's graph and plan with ``tools``."""
import glob
import time
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import APPENDS, PREFILL_END, TOOLS, TURN_IDS, WalkStep
from mstar.model.components.python_tool import (
    AGENT_ANSWER,
    AGENT_PROMPTS,
    ANSWER,
    SPAWNED,
    TOOL_CALLS,
    TOOL_HANDLERS,
    TOOL_TEXT,
    TOOLS_EXIT,
    QwenToolFormat,
    ToolRouterSubmodule,
    ToolSubmodule,
    _unpack,
    run_python,
)
from mstar.model.keys import SPEAK_TEXT
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

CALLS = (
    "<tool_call>\n<function=python>\n<parameter=code>\nprint(6 * 7)\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=python>\n<parameter=code>\nprint('b')\n</parameter>\n</function>\n</tool_call>"
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    found = glob.glob("/workspace/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/")
    if not found:
        pytest.skip("Qwen3.5-0.8B's tokenizer is not in the HF cache")
    return AutoTokenizer.from_pretrained(found[0], local_files_only=True)


def test_parallel_calls_are_parsed_in_order():
    calls = QwenToolFormat().parse("thinking aloud\n\n" + CALLS)
    assert [(c.name, c.arguments) for c in calls] == [
        ("python", {"code": "print(6 * 7)"}), ("python", {"code": "print('b')"}),
    ]


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_the_continuation_is_what_the_chat_template_writes_after_the_calls(tokenizer, enable_thinking):
    fmt = QwenToolFormat()
    user = [{"role": "user", "content": "go"}]
    calls = [{"function": {"name": "python", "arguments": {"code": "print(6 * 7)"}}},
             {"function": {"name": "python", "arguments": {"code": "print('b')"}}}]
    reply = [{"role": "assistant", "content": "", "tool_calls": calls}]
    answers = [{"role": "tool", "content": "42\n"}, {"role": "tool", "content": "b\n"}]
    kwargs = dict(tokenize=False, enable_thinking=enable_thinking)
    whole = tokenizer.apply_chat_template(user + reply + answers, add_generation_prompt=True, **kwargs)
    opening, closing = fmt.render_frame(enable_thinking)
    # (the template writes the calls after an emptied think block, which the model's own reasoning
    # replaces in the request's state; what follows the calls is the same)
    assert whole.endswith(CALLS + "<|im_end|>" + opening + fmt.wrap("42\n") + fmt.wrap("b\n") + closing)


def test_a_run_prints_stops_at_its_time_and_reports_a_crash():
    assert run_python("print(6 * 7)") == "42\n"
    started = time.monotonic()
    assert "ran past 0.5 seconds" in run_python("import time; time.sleep(30)", timeout=0.5)
    assert time.monotonic() - started < 5
    assert "ZeroDivisionError" in run_python("1 / 0")


def _turn(tokenizer, text):
    stop = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False) + [stop])


def _routed(tokenizer, text, speak=False):
    """The router's output for a turn, and the exit it names."""
    router = ToolRouterSubmodule(tokenizer)
    ids = _turn(tokenizer, text)
    row = {"ids": ids, "start": 100, "speak": speak, "spawn_agent": False, "child": False, "thinking": False}
    out = router.forward_batched("route", SimpleNamespace(request_ids=["r"]), rows=[row])["r"]
    return ids, out[TOOL_CALLS][0], router.check_stop("r", None, out).exit, out


def _ran(tokenizer, calls, ids, budget=200, handlers=None):
    tool = ToolSubmodule(tokenizer, handlers or TOOL_HANDLERS)
    row = {"payload": _unpack(calls), "enable_thinking": False, "budget": budget, "answers": None}
    return tool.forward_batched("tool", SimpleNamespace(request_ids=["r"]), rows=[row])["r"][TOOL_TEXT]


def test_the_router_sends_a_turn_with_calls_to_the_tools_and_one_without_to_its_answer(tokenizer):
    ids, calls, exit_name, _ = _routed(tokenizer, CALLS)
    assert exit_name == TOOLS_EXIT
    assert _unpack(calls) == {"tail": ids[-1].item(), "start": 100, "calls": [
        ["python", {"code": "print(6 * 7)"}], ["python", {"code": "print('b')"}]]}
    assert _routed(tokenizer, "The answer is 42.")[2] == ANSWER
    assert SPEAK_TEXT not in _routed(tokenizer, CALLS, speak=True)[3]
    # the stop token is fed by the step after it: the turn's last id belongs at the prefill's end plus the rest
    row = ToolRouterSubmodule(tokenizer).prepare_inputs(
        "route", SimpleNamespace(step_metadata={}),
        {"new_token": [ids[:1]], TURN_IDS: [ids[1:]], PREFILL_END: [torch.tensor([95])]})
    assert row.kwargs["start"] == 95 + ids.numel() - 1


def test_the_tool_node_runs_every_call_by_its_handler_and_writes_the_next_prefills_text(tokenizer):
    ids, calls, *_ = _routed(tokenizer, CALLS)
    text, start = _ran(tokenizer, calls, ids)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # the ids to write start with the stop token the decoder may drop, and the position it belongs at
    assert text[0].item() == im_end and start.tolist() == [100]
    written = tokenizer.decode(text[1:].tolist())
    assert "<tool_response>\n42\n</tool_response>" in written and "<tool_response>\nb\n</tool_response>" in written
    assert written.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_a_call_to_a_tool_nobody_handles_is_an_error_the_model_reads_and_a_spent_budget_leaves_no_room(tokenizer):
    text = "<tool_call>\n<function=ghost>\n</function>\n</tool_call>"
    ids, calls, exit_name, _ = _routed(tokenizer, text)
    assert exit_name == TOOLS_EXIT
    written = tokenizer.decode(_ran(tokenizer, calls, ids)[0][1:].tolist())
    assert "there is no tool 'ghost'" in written
    ids, calls, *_ = _routed(tokenizer, CALLS)
    assert _ran(tokenizer, calls, ids, budget=5)[0].tolist() == [ids[-1].item()]


SAY = '<tool_call>\n<function=speak>\n<parameter=text>\nIt is 42.\n</parameter>\n</function>\n</tool_call>'


def test_a_turn_that_only_speaks_exits_through_speak_and_writes_its_text_to_the_speech_edge(tokenizer):
    _, _, exit_name, out = _routed(tokenizer, SAY, speak=True)
    assert exit_name == "speak" and _unpack(out[SPEAK_TEXT][0]) == {"text": "It is 42.", "voice": None}
    # nothing on the speech edge when the request has no audio output, or the turn also calls a tool
    for text, speak in ((SAY, False), (CALLS + SAY, True)):
        _, _, exit_name, out = _routed(tokenizer, text, speak=speak)
        assert exit_name == TOOLS_EXIT and SPEAK_TEXT not in out


ASKS = (
    '<tool_call>\n<function=spawn_agent>\n<parameter=task>\nsum 1..10\n</parameter>\n</function>\n</tool_call>'
    '<tool_call>\n<function=spawn_agent>\n<parameter=task>\nsum 1..200\n</parameter>\n</function>\n</tool_call>')


def _spawner(tokenizer, **flags):
    router = ToolRouterSubmodule(tokenizer, render_agent=lambda task, thinking: [len(task), int(thinking)])
    ids = _turn(tokenizer, ASKS)
    row = {"ids": ids, "start": 100, "speak": False, "spawn_agent": True, "child": False, "thinking": False, **flags}
    out = router.forward_batched("route", SimpleNamespace(request_ids=["r"]), rows=[row])["r"]
    return router, out


def test_a_turn_that_only_spawns_agents_persists_a_prompt_per_task_and_the_spawned_flag(tokenizer):
    router, out = _spawner(tokenizer)
    assert router.check_stop("r", None, out).exit == "spawn"
    assert [out[name][0].tolist() for name in AGENT_PROMPTS if name in out] == [[9, 0], [10, 0]]
    assert out[SPAWNED][0].tolist() == [1]
    # once per request, and never in an agent: the same turn is calls nobody handles
    for flags in ({"spawn_agent": False}, {"child": True, "spawn_agent": False}):
        router, out = _spawner(tokenizer, **flags)
        assert router.check_stop("r", None, out).exit == TOOLS_EXIT and AGENT_PROMPTS[0] not in out


def test_an_agent_that_answers_persists_its_final_turn_for_its_parent_and_a_parent_does_not(tokenizer):
    router = ToolRouterSubmodule(tokenizer)
    ids = _turn(tokenizer, "The sum is 55.")
    for child in (True, False):
        row = {"ids": ids, "start": 100, "speak": False, "spawn_agent": False, "child": child, "thinking": False}
        out = router.forward_batched("route", SimpleNamespace(request_ids=["r"]), rows=[row])["r"]
        assert (AGENT_ANSWER in out) is child
        assert router.check_stop("r", None, out).exit == ANSWER
    assert out is not None and ids.tolist() == _turn(tokenizer, "The sum is 55.").tolist()


def test_the_join_frames_each_agents_answer_as_a_tool_response_in_task_order_with_errors_in_place(tokenizer):
    tool = ToolSubmodule(tokenizer, TOOL_HANDLERS)
    _, calls, *_ = _routed(tokenizer, ASKS)
    metadata = {"agents": 3, "agent_errors": {"1": "Error: the agent timed out"}, "enable_thinking": False}
    answers = [_turn(tokenizer, "55"), _turn(tokenizer, "210")]
    row = tool.prepare_inputs(
        "tool", SimpleNamespace(step_metadata=metadata, max_tokens=500),
        {TOOL_CALLS: [calls], "agent_answers": answers}).kwargs
    assert row["answers"] == ["55", "Error: the agent timed out", "210"]
    text, start = tool.forward_batched("tool", SimpleNamespace(request_ids=["r"]), rows=[row])["r"][TOOL_TEXT]
    written = tokenizer.decode(text[1:].tolist())
    bodies = [part.split("\n</tool_response>")[0].strip() for part in written.split("<tool_response>")[1:]]
    assert bodies == ["55", "Error: the agent timed out", "210"] and start.tolist() == [100]
    # a step that joins nothing runs the calls' handlers
    assert tool.prepare_inputs(
        "tool", SimpleNamespace(step_metadata={}, max_tokens=500), {TOOL_CALLS: [calls]}).kwargs["answers"] is None


def _model():
    model = object.__new__(Qwen3_5DenseModel)
    model.config, model.media, model.tools = SimpleNamespace(max_position_embeddings=64), (), ("python",)
    return model


def test_a_model_with_the_tool_has_its_walk_its_cycle_and_a_reserve_that_is_the_budget():
    model = _model()
    walks = model.get_graph_walk_graphs()
    assert {"route", "tool", "decode", "prefill_text"} <= set(walks)
    assert [e.name for e in walks["decode"].accumulated_outputs] == [TURN_IDS]
    assert set(walks["route"].get_nodes()["Router"].input_names) == {"new_token", TURN_IDS, PREFILL_END}
    assert set(walks["tool"].get_nodes()["Tool"].input_names) == {TOOL_CALLS}

    plan = model.walk_plan("LLM", ["text"], ["text"], {"text_inputs": [object()]}, {})
    assert [step.walk for step in plan] == ["prefill_text", "decode", "route", "tool", "prefill_text"]
    assert plan[2].exits == {"answer": 5, "speak": 5, "tools": 3} and plan[4].exits == {"": 1}
    assert all(step.step_metadata.get(TOOLS) for step in (plan[0], plan[1], plan[4]))
    # the continuation writes text no rebuild replays, so its request is not dropped (`Rebuilds.note_appended`)
    assert plan[4].step_metadata[APPENDS] and not plan[0].step_metadata.get(APPENDS)
    assert model.get_loop_caps(max_output_tokens=100) == {"decode_loop": 101}

    # speaking is a tool with no handler: the template offers it, and the plan tells the router a speech edge exists
    model.tools = ("python", "speak")
    assert model.walk_plan("LLM", ["text"], ["text", "audio"], {"text_inputs": [object()]}, {})[2].step_metadata[
        "speak"] is True
    assert model.walk_plan("LLM", ["text"], ["text"], {"text_inputs": [object()]}, {})[2].step_metadata[
        "speak"] is False

    # a deployment that spawns: the join and a second continuation follow, an agent's plan has neither
    model.tools, model.spawn = ("python", "spawn_agent"), True
    plan = model.walk_plan("LLM", ["text"], ["text"], {"text_inputs": [object()]}, {})
    assert [step.walk for step in plan] == [
        "prefill_text", "decode", "route", "tool", "prefill_text", "tool", "prefill_text"]
    assert plan[2].exits == {"answer": 7, "speak": 7, "tools": 3, "spawn": 5} and plan[2].spawns == AGENT_PROMPTS
    assert plan[5].joins and not plan[3].joins and plan[6].exits == {"": 1}
    child = model.walk_plan("LLM", ["text"], ["text"], {"text_inputs": [object()]}, {"agent_child": "parent"})
    assert len(child) == 5 and not child[2].spawns and child[2].step_metadata["agent_child"] is True
    assert {"agent_0", "spawned", "agent_answer"} <= {
        e.name for e in model.get_graph_walk_graphs()["route"].outputs}
    model.tools, model.spawn = ("python",), False

    model.tools = ()
    assert "tool" not in model.get_graph_walk_graphs()
    assert [step.walk for step in model.walk_plan("LLM", ["text"], ["text"], {"text_inputs": [object()]}, {})] == [
        "prefill_text", "decode"]
    assert isinstance(model.walk_plan("LLM", ["text"], ["text"], {"text_inputs": [object()]}, {})[0], WalkStep)


def test_the_tool_deployment_places_the_tool_node_on_a_rank_of_its_own():
    from pathlib import Path

    import yaml

    from mstar.validation.placement import check_placement

    path = Path(__file__).resolve().parents[2] / "configs" / "qwen3_5_0.8b_tool.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["model_kwargs"] == {"tools": ["python"]}
    assert check_placement(_model(), config) == []
    # a deployment that turns the tool on must place its node
    config["node_groups"] = config["node_groups"][:1]
    with pytest.raises(ValueError, match="needs the nodes"):
        _model().validate_config_yaml(config, str(path))
