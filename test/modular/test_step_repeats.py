"""Which decode steps may stand for several iterations (``step_repeats:``)."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.device_run import RunState
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import HostRows
from mstar.testing_utils.worker import make_worker
from mstar.worker.services import _step_repeats  # noqa: PLC2701
from mstar.worker.step_repeats import RepeatConfig, StepRepeats, loop_signals, parse_step_repeats
from mstar.worker.worker import Worker

NODE, WALK = "decoder", "decode"


def _loop(*outputs: GraphEdge) -> Loop:
    edges = [
        GraphEdge(next_node=NODE, name="text_inputs"), GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token"), *outputs,
    ]
    node = GraphNode(name=NODE, input_names=["text_inputs"], outputs=edges)
    return Loop(name="decode_loop", section=node, max_iters=8, outputs=[])


@pytest.mark.parametrize(("block", "match"), [
    ({NODE: {"max": 0}}, "at least 1"),
    ({NODE: {"max": 8, "auto": True}}, "not settings"),
    ({NODE: {"max": 1, "ready_waits": True}}, "needs max"),
    ({NODE: {"max": 1, "device_stop": True}}, "needs max"),
    ({"other": {"max": 8}}, "not nodes"),
])
def test_a_setting_that_cannot_run_is_refused_at_load(block, match):
    with pytest.raises(ValueError, match=match):
        parse_step_repeats(block, {NODE})


def test_a_lockstep_node_is_refused_before_any_rank_launches():
    model = SimpleNamespace(get_graph_walk_graphs=lambda: {WALK: _loop()})
    with pytest.raises(ValueError, match="lockstep"):
        _step_repeats(model, {NODE: {"max": 8}}, {(NODE, WALK)}, {NODE}, ())


def test_a_per_iteration_output_other_than_the_client_token_refuses_its_walk():
    good = loop_signals(_loop(), NODE, ())
    assert good.refusal is None and good.loop_back == {"text_inputs"}
    bad = loop_signals(_loop(GraphEdge(next_node="vocoder", name="frames")), NODE, ())
    assert "read once an iteration" in bad.refusal


def test_a_node_the_config_names_with_no_walk_that_can_repeat_stops_the_worker():
    repeats = StepRepeats({NODE: RepeatConfig(max=8)}, {(NODE, WALK): loop_signals(_loop(), NODE, ())})
    with pytest.raises(ValueError, match="no walk of it can"):
        repeats.settle(lambda node, walk: "resource 'attn' allows one iteration a step")


@pytest.mark.parametrize(("ready_waits", "repeats", "owed", "want"), [
    (True, 1, False, True),   # the run the waiting rows are owed
    (True, 1, True, False),   # once: a step narrowed to 1 can't keep them out
    (True, 8, False, False),  # it just ran long
    (False, 1, False, False),  # the policy is off: today's behaviour
])
def test_ready_rows_wait_only_for_the_one_run_owed(ready_waits, repeats, owed, want):
    config = RepeatConfig(max=8, ready_waits=ready_waits)
    assert StepRepeats({NODE: config}, {}).owes_run(NODE, repeats, owed) is want


def test_a_device_run_starts_with_only_real_rows_live_and_keeps_a_column_an_iteration():
    state = RunState(4, 3, {2}, torch.device("cpu"))
    state.stage(2, [False, True], limit=3)
    assert state.marks.tolist() == [False, False, True, True], "padding rows start stopped"
    assert (state.continuation.item(), state.limit.item(), state.immune.tolist()) == (1, 3, [False, True, False, False])

    state.tokens[:] = 0
    for token in ([[5], [6], [0], [0]], [[7], [8], [0], [0]]):
        state.keep_token(torch.tensor(token).reshape(-1))
        state.index += 1
    results = state.results(3, ("new_token",))
    assert results["new_token"][:2, :2].tolist() == [[5, 7], [6, 8]]
    assert state.index.item() == 2


def test_a_device_run_that_ended_short_settles_its_width_before_anything_reads_it():
    """A run ended short wrote only the columns it ran, so the host cuts them
    to that width and reads the step as the width it really had."""
    rows = HostRows((1, 2), {"new_token": torch.tensor([[5, 9, 0], [6, 2, 0]])})
    ctx = SimpleNamespace(repeats=3)
    batch_n = SimpleNamespace(
        batch=SimpleNamespace(repeats=3), node_batch=SimpleNamespace(step_context=ctx), graph_walk=WALK,
    )
    outputs = SimpleNamespace(repeats=3, run_replays=torch.tensor([2]))

    Worker._settle_device_run(batch_n, outputs, rows)

    assert rows.buffers["new_token"].tolist() == [[5, 9], [6, 2]]
    assert (batch_n.batch.repeats, ctx.repeats, outputs.repeats) == (2, 2, 2)


def test_a_step_that_admitted_before_the_main_thread_looked_still_reads_its_narrowed_count():
    """The launch event already set is the usual case at a short admit."""
    event = threading.Event()
    event.set()
    pending = SimpleNamespace(
        batch=SimpleNamespace(repeats=8),
        node_batch=SimpleNamespace(launch_started_event=event, step_context=SimpleNamespace(repeats=1)),
    )

    make_worker(_step_repeats=StepRepeats({}, {}))._await_admit_settled(pending)

    assert pending.batch.repeats == 1
