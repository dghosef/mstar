"""Qwen3.5 text prefill and decode rows ride in one step: the model declares
the combined walk, a mixed step tracks only its prefill rows' prompt tokens,
and the GDN kernel choice follows a packed capture rather than the spans."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.linear_attn.gdn import GDNManager
from mstar.model.keys import MIXED
from mstar.model.qwen3_5.config import SAMPLER
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs


def test_text_prefill_and_decode_share_the_mixed_walk():
    assert Qwen3_5DenseModel.get_combined_graph_walks(object.__new__(Qwen3_5DenseModel)) == {
        "LLM": {MIXED: {"prefill_text", "decode"}},
    }


def test_a_mixed_step_tracks_only_its_prefill_rows_prompt_tokens():
    llm = object.__new__(LLMSubmodule)
    prompt = torch.tensor([5, 6, 7])
    rows = [
        ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([9]), graph_walk="decode"),
        ARNodeInputs(input_seq_len=3, input_ids=prompt, graph_walk="prefill_text"),
    ]

    step = llm.declare_step(MIXED, [0, 1], rows)

    tracked = step.steps[SAMPLER].prefill_tracked_tokens
    assert list(tracked) == [1] and tracked[1] is prompt


class _Recorder(GDNManager):
    def __init__(self):
        self.kinds = []
        self.config = SimpleNamespace(splits=())
        self._splits = {}

    def _get_wrapper(self, label, is_decode, lease=None, split=False):
        self.kinds.append("decode" if is_decode else "prefill")
        return SimpleNamespace(plan=lambda *args: None)


@pytest.mark.parametrize("force_prefill,spans,kind", [
    (False, [1, 1], "decode"),
    (False, [1, 3], "prefill"),
    (True, [1, 1], "prefill"),  # a packed capture replays mixed steps
])
def test_the_kernel_follows_a_packed_capture_not_the_spans(force_prefill, spans, kind):
    gdn = _Recorder()
    segments = [SimpleNamespace(span=s) for s in spans]
    addressing = SimpleNamespace(slot_indices=torch.zeros(4, dtype=torch.long),
                                 has_state=torch.zeros(4, dtype=torch.bool),
                                 snapshots=(), snapshot_every=0)
    ctx = SimpleNamespace(force_prefill=force_prefill, slot_lease=None, graph_walk="decode")

    gdn._build_plan("main", segments, addressing, ctx)

    assert gdn.kinds == [kind]


@pytest.mark.parametrize("walk,inputs,length", [
    ("decode", {}, 1),
    ("prefill_text", {"text_inputs": [torch.zeros(7, dtype=torch.long)]}, 7),
    ("prefill_vision", {"text_inputs": [torch.zeros(7, dtype=torch.long)]}, None),
])
def test_text_and_decode_rows_report_their_length(walk, inputs, length):
    info = object.__new__(LLMSubmodule).get_input_sequence_len(walk, _info(walk), inputs)

    assert (info.seq_len if info is not None else None) == length


def test_text_prefill_chunks_under_a_budget_that_fits_the_captures():
    model = Qwen3_5DenseModel.__new__(Qwen3_5DenseModel)
    model.config = SimpleNamespace(max_position_embeddings=4096)
    model.vision_config = None
    policies = model.chunking_policy_of()

    assert policies[("LLM", "prefill_text")].chunkable
    assert ("LLM", "prefill_vision") not in policies
    mixed = policies[("LLM", MIXED)]
    assert not mixed.chunkable and mixed.max_batch_tokens == LLMSubmodule.MAX_BATCH_TOKENS
    # decode rows count inside the budget, and a step over a bucket pads to the next one up
    assert LLMSubmodule.MAX_BATCH_TOKENS in LLMSubmodule.PREFILL_TOKEN_BUCKETS



class _State:
    """What a request's position counter and KV hold, as the engine commits steps."""

    def __init__(self):
        self.tokens: list[int] = []

    def position(self, rid, label):
        return len(self.tokens)


def _decoder(state):
    llm = object.__new__(LLMSubmodule)
    torch.nn.Module.__init__(llm)
    llm.node_resources, llm.request_states = {llm.POS_KEY: state}, {}
    return llm


def _info(walk, **metadata):
    from mstar.conductor.request_info import CurrentForwardPassInfo
    return CurrentForwardPassInfo(
        request_id="wire", rid_handle="r", graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=64, step_metadata=metadata,
    )


@pytest.mark.parametrize("speculated_step_ran", [False, True])
@pytest.mark.parametrize("ran_before_prepare", [False, True])
def test_a_prefill_after_a_decode_lands_after_the_stop_token_whether_or_not_the_speculated_step_ran(
    speculated_step_ran, ran_before_prepare,
):
    """A decode loop stopped by a token has its next iteration launched already, and that step's row is dropped
    only after it ran: the stop token is in the request's state if it did and is not if it did not. The row is
    prepared on a host thread that may run before that step commits, so the prefill decides when it is declared,
    in step order: the same ids land either way. A plain request's row is not touched."""
    from mstar.conductor.request_info import FINAL_PREFILL, PREFILL_END, TOOLS

    prompt, turn, tail = [11, 12, 13], [21, 22, 23, 99], [31, 32]      # 99 = <|im_end|>, the stop token
    state = _State()
    llm = _decoder(state)

    state.tokens += prompt                                              # the prefill commits the prompt
    outputs = {"new_token": [torch.tensor([turn[0]])]}
    llm.postprocess("wire", _info("prefill_text", **{TOOLS: True, FINAL_PREFILL: True}), outputs)
    (end,) = outputs[PREFILL_END]
    assert end.tolist() == [len(prompt)]
    state.tokens += turn[:-1]                                           # a decode step feeds each token but the stop

    def speculated_step():
        if speculated_step_ran:
            state.tokens.append(turn[-1])                               # the one speculated past the stop

    # what the router and tool node hand over: the ids to write from the stop token on, and where it belongs
    row = [torch.tensor([turn[-1], *tail]), torch.tensor([int(end) + len(turn) - 1])]
    info = _info("prefill_text", **{TOOLS: True})
    if ran_before_prepare:
        speculated_step()
    inp = llm.prepare_inputs("prefill_text", info, {"text_inputs": row})
    assert llm.get_input_sequence_len("prefill_text", info, {"text_inputs": row}).seq_len == len(inp.input_ids)
    if not ran_before_prepare:
        speculated_step()                                               # it commits after the row was prepared
    llm.declare_step("prefill_text", ["r"], [inp])
    llm.declare_step("prefill_text", ["r"], [inp])                      # a retried declaration drops nothing more
    assert inp.input_seq_len == inp.input_ids.numel()
    state.tokens += inp.input_ids.tolist()

    assert state.tokens == prompt + turn + tail
    plain = llm.prepare_inputs("prefill_text", _info("prefill_text"), {"text_inputs": [torch.tensor([1, 2, 3])]})
    llm.declare_step("prefill_text", ["r"], [plain])
    assert plain.input_ids.tolist() == [1, 2, 3] and plain.input_seq_len == 3


def test_a_prefill_after_a_decode_that_finds_its_request_off_by_more_than_the_speculated_step_fails():
    state = _State()
    llm = _decoder(state)
    state.tokens += [4, 5, 6]                       # three ids past where the row says its first belongs
    inp = llm.prepare_inputs(
        "prefill_text", _info("prefill_text"), {"text_inputs": [torch.tensor([99, 31]), torch.tensor([0])]})
    with pytest.raises(AssertionError, match="already written"):
        llm.declare_step("prefill_text", ["r"], [inp])


def test_a_prefill_that_is_not_the_last_before_decode_routes_no_token_with_or_without_tools():
    from mstar.conductor.request_info import FINAL_PREFILL, PREFILL_END, TOOLS

    llm = _decoder(_State())
    for metadata in ({FINAL_PREFILL: False}, {FINAL_PREFILL: False, TOOLS: True}):
        outputs = {"new_token": [torch.tensor([7])], "text_inputs": [torch.tensor([7])]}
        llm.postprocess("wire", _info("prefill_text", **metadata), outputs)
        assert not outputs
    outputs = {"new_token": [torch.tensor([7])]}
    llm.postprocess("wire", _info("prefill_text", **{FINAL_PREFILL: True}), outputs)
    assert PREFILL_END not in outputs and outputs["new_token"]


def test_a_tool_requests_decode_accumulates_its_tokens_and_a_plain_one_does_not():
    from mstar.conductor.request_info import TOOLS, TURN_IDS

    llm = _decoder(_State())
    for metadata, expected in (({TOOLS: True}, True), ({}, False)):
        outputs = {"new_token": [torch.tensor([7])]}
        llm.postprocess("wire", _info("decode", **metadata), outputs)
        assert (TURN_IDS in outputs) == expected


def test_only_a_deployment_with_tools_changes_its_decode_stop_policy():
    from mstar.conductor.request_info import TOOLS
    from mstar.model.submodule_base import StopKind, TokenStop

    llm = _decoder(_State())
    llm.token_stop = TokenStop(
        loop="decode_loop", buffer="new_token", stop_ids=frozenset({99}), sampler="sampler", walk="decode",
        runtime_check=True)
    # a node that runs no tool rounds: the EOS-only policy, with no iteration limit, as ever
    llm.tool_budget = False
    (policy,) = llm.get_stop_policies("decode").values()
    assert policy.kind is StopKind.EOS and policy.limits(_info("decode", **{TOOLS: True})).max_iters is None
    # with tools, a tool request's decode runs under what is left of its budget, a plain one under its cap
    llm.tool_budget = True
    (policy,) = llm.get_stop_policies("decode").values()
    assert policy.limits(_info("decode", **{TOOLS: True})).max_iters == 64
    assert policy.limits(_info("decode")).max_iters > 10 ** 9


def test_the_rows_of_a_prefill_that_is_not_the_last_before_decode_have_no_real_sample():
    from types import SimpleNamespace

    from mstar.conductor.request_info import FINAL_PREFILL
    from mstar.model.submodule_base import TokenStop
    from mstar.testing_utils.engine import make_engine, make_mgmt

    stop = TokenStop(
        loop="decode_loop", buffer="new_token", stop_ids=frozenset({99}), sampler="sampler", walk="decode",
        runtime_check=True, prefill_walks=frozenset({"prefill_text"}), last_prefill_flag=FINAL_PREFILL)
    eng = make_engine(_submodules={"LLM": make_mgmt(SimpleNamespace(token_stop=stop))})
    infos = {1: _info("prefill_text", **{FINAL_PREFILL: False}), 2: _info("prefill_text", **{FINAL_PREFILL: True}),
             3: _info("decode"), 4: _info("prefill_text")}

    def batch(rids, walk, request_walks=None):
        return SimpleNamespace(
            node_name="LLM", request_ids=rids, per_request_info_wrapped=infos,
            step_context=SimpleNamespace(graph_walk=walk, request_walks=request_walks))

    # a step of one walk, and a mixed step; a step of decode rows is not looked into
    assert eng.unreal_sample_rids(batch([1, 2], "prefill_text")) == {1}
    assert eng.unreal_sample_rids(batch([1, 3, 4], "mixed", request_walks={1: "prefill_text"})) == {1}
    assert eng.unreal_sample_rids(batch([3], "decode")) == set()
