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
    llm.tool_rounds = True
    return llm


def _info(walk, **metadata):
    from mstar.conductor.request_info import CurrentForwardPassInfo
    return CurrentForwardPassInfo(
        request_id="wire", rid_handle="r", graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=64, step_metadata=metadata,
    )


@pytest.mark.parametrize("speculated_step_ran", [False, True])
def test_a_continuation_lands_after_the_stop_token_whether_or_not_the_speculated_step_ran(speculated_step_ran):
    """A decode loop stopped by a token has its next iteration launched already, and that step's row is
    dropped only after it ran: the stop token is in the request's state if it did and is not if it did not.
    The continuation prefill writes the same ids either way, a single prefill of the whole sequence."""
    from mstar.conductor.request_info import CONTINUES, TOOLS

    prompt, turn, tail = [11, 12, 13], [21, 22, 23, 99], [31, 32]      # 99 = <|im_end|>, the stop token
    state = _State()
    llm = _decoder(state)

    state.tokens += prompt                                              # the prefill commits the prompt
    llm.postprocess("wire", _info("prefill_text", **{TOOLS: True}), {})
    state.tokens += turn[:-1]                                           # a decode step feeds each token but the stop
    if speculated_step_ran:
        state.tokens.append(turn[-1])                                   # the one speculated past the stop

    row = torch.tensor([len(turn), turn[-1], *tail])
    info = _info("prefill_text", **{TOOLS: True, CONTINUES: True})
    ids = llm._text_ids(info, {"text_inputs": [row]})
    assert llm.get_input_sequence_len("prefill_text", info, {"text_inputs": [row]}).seq_len == len(ids)
    state.tokens += ids.tolist()

    assert state.tokens == prompt + turn + tail


def test_a_continuation_that_finds_its_request_off_by_more_than_the_speculated_step_is_refused():
    from mstar.conductor.request_info import CONTINUES, TOOLS

    state = _State()
    llm = _decoder(state)
    state.tokens += [1, 2, 3]
    llm.postprocess("wire", _info("prefill_text", **{TOOLS: True}), {})
    state.tokens += [4, 5, 6]                       # three steps past a one-token turn
    row = torch.tensor([1, 99, 31])
    with pytest.raises(ValueError, match="past where its 1-token turn ends"):
        llm._text_ids(_info("prefill_text", **{TOOLS: True, CONTINUES: True}), {"text_inputs": [row]})


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
    llm.tool_rounds = False
    (policy,) = llm.get_stop_policies("decode").values()
    assert policy.kind is StopKind.EOS and policy.limits(_info("decode", **{TOOLS: True})).max_iters is None
    # with tools, a tool request's decode runs under what is left of its budget, a plain one under its cap
    llm.tool_rounds = True
    (policy,) = llm.get_stop_policies("decode").values()
    assert policy.limits(_info("decode", **{TOOLS: True})).max_iters == 64
    assert policy.limits(_info("decode")).max_iters > 10 ** 9
