"""What a request generates is keyed the same way its prompt was.

In agent traffic one turn's output is the next turn's prompt, so a generated
page is worth as much as a prompt page — but the tokens only exist once they are
sampled, and the only host copy of them is the one the stop check already takes.
That copy is early enough: a step writes the id the step before it sampled, so by
the commit that fills a page's last slot, every id in that page is already on the
host and the page is indexed there and then.

The page where the prompt ends and the generation begins belongs to both. Its
key covers the prompt's trailing tokens and the first generated ones together,
which is why the request carries that tail: the manager sees sampled tokens, and
nothing else of the prompt.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import _MeasuredRow
from mstar.engine.resources.base import KeyedWalks
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs

PAGE_SIZE = 128
ROOT = b"a root"
NODE = "LLM"
WALK = "decode"
TENSOR = "text_inputs"
EOS = 2


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int = 64) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=8192,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT)
    return kv


def _pages(tokens: list[int]) -> list[list[int]]:
    return [
        tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
    ]


def _layout(prompt: list[int]) -> StreamLayout:
    return StreamLayout(
        spans=(PrefixSpan(len(prompt), len(prompt), WALK),), growth=(("decode_loop", 1),), keyed_spans=1,
    )


def _ingest(kv: KVManager, rid: str, prompt: list[int], chains: bool = True, decode: int = 0):
    whole = len(prompt) // PAGE_SIZE
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain(_pages(prompt))},
        prefix_tail={"main": prompt[whole * PAGE_SIZE:]},
        prefix_decode={"main": TENSOR} if chains else None,
        kv_layout={"main": _layout(prompt)},
        loop_caps={"decode_loop": decode},
    ))


def _step(kv: KVManager, rid: str, span: int) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(
        request_ids=(rid,), graph_walk=WALK, slot=0, capture=False,
    )
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _sampled(kv: KVManager, rid: str, token: int) -> None:
    """The host copy of what a step sampled, taken after its stop check."""
    kv.extend_prefix_chain(
        rid, NODE, WALK, {TENSOR: [torch.tensor([token])]},
    )


def _prefill(kv: KVManager, rid: str, prompt: list[int], sampled: int = 9000):
    """The prompt, and the id the prefill step sampled from it."""
    _step(kv, rid, len(prompt))
    _sampled(kv, rid, sampled)


def _decode(kv: KVManager, rid: str, token: int) -> None:
    """One decode step: it writes the id sampled before it, and samples ``token``."""
    _step(kv, rid, 1)
    _sampled(kv, rid, token)


def _indexed(kv: KVManager) -> int:
    return len(kv._index.pages())


# ── the page the generation fills ───────────────────────────────────────


def test_a_generated_page_is_indexed_at_the_commit_that_fills_it():
    kv = _manager()
    prompt = list(range(PAGE_SIZE))
    _ingest(kv, "r0", prompt, decode=PAGE_SIZE + 1)
    _prefill(kv, "r0", prompt)
    assert _indexed(kv) == 1, "the prompt's own page was not indexed"

    seen = []
    for token in range(1, PAGE_SIZE + 2):
        _decode(kv, "r0", 9000 + token)
        seen.append((kv._streams["r0"]["main"].stored_len, _indexed(kv)))

    filled = [step for step, (stored, _) in enumerate(seen) if stored == 2 * PAGE_SIZE]
    assert len(filled) == 1, (
        "the stream passed two pages of tokens, so the step that filled the "
        "second one is not the step under test"
    )
    at = filled[0]
    assert seen[at - 1][1] == 1, "a page was indexed before its last slot was written"
    assert seen[at][1] == 2, (
        "the page was not indexed at the commit that filled it; every id in it "
        "was already on the host by then"
    )
    kv.assert_pages_conserved()


def test_a_generated_page_is_matched_by_the_request_that_asks_for_it_next():
    kv = _manager()
    prompt = list(range(100))
    generated = list(range(9000, 9000 + 28))
    _ingest(kv, "r0", prompt, decode=len(generated))
    _prefill(kv, "r0", prompt, sampled=generated[0])
    for token in generated[1:]:
        _decode(kv, "r0", token)
    # the step that writes the last generated id, which fills the page
    _decode(kv, "r0", 9999)

    # the next turn: the whole of the first turn is now the prompt
    next_prompt = prompt + generated + list(range(200, 260))
    _ingest(kv, "r1", next_prompt)

    inputs = ARNodeInputs(input_seq_len=len(next_prompt))
    assert kv.resolve_cached_prefix("r1", NODE, WALK, inputs).tokens == PAGE_SIZE, (
        "the page the first turn generated was not there for the second"
    )
    kv.assert_pages_conserved()


def test_chain_key_is_the_key_the_index_files_each_page_under():
    """A recurrent pool files its snapshots under `chain_key`, so a key the index doesn't use
    names a snapshot no probe finds. The key the probe rooted for the prompt's partial last
    page becomes such a key once the generation fills that page."""
    kv = _manager()
    prompt = list(range(200))
    generated = list(range(9000, 9000 + 56))
    _ingest(kv, "r0", prompt, decode=len(generated))
    kv.resolve_cached_prefix("r0", NODE, WALK, ARNodeInputs(input_seq_len=len(prompt)))
    _prefill(kv, "r0", prompt, sampled=generated[0])
    for token in generated[1:]:
        _decode(kv, "r0", token)
    _decode(kv, "r0", 9999)
    assert _indexed(kv) == 2
    for pages in (1, 2):
        assert kv._index.page_for(kv.chain_key("r0", "main", pages)) is not None, f"page {pages} is keyed apart"
    assert kv.chain_keys("r0", "main", 3) == [kv.chain_key("r0", "main", 1), kv.chain_key("r0", "main", 2)]


# ── nodes that cannot chain what they generate ──────────────────────────


def test_a_stop_on_the_boundary_leaves_the_page_one_slot_short():
    kv = _manager()
    prompt = list(range(100))
    _ingest(kv, "r0", prompt, decode=27)
    _prefill(kv, "r0", prompt, sampled=9000)

    # the request stops here: its last two sampled ids, the one before the
    # stop and the stop itself, never reach a step that could write them
    for token in range(1, 28):
        _decode(kv, "r0", 9000 + token)
    _sampled(kv, "r0", EOS)

    stream = kv._streams["r0"]["main"]
    assert stream.chain.keyed_pages == 1, "the chain did not reach the page boundary"
    assert stream.stored_len == PAGE_SIZE - 1, (
        "the id that ended the request was written into a slot"
    )
    assert _indexed(kv) == 0, (
        "a page whose last slot holds nothing was offered to the index"
    )
    kv.assert_pages_conserved()


def test_a_node_whose_decode_ids_are_not_the_sampled_token_stops_at_its_prompt():
    kv = _manager()
    prompt = list(range(PAGE_SIZE))
    _ingest(kv, "r0", prompt, chains=False, decode=PAGE_SIZE + 1)
    _prefill(kv, "r0", prompt)
    after_prompt = _indexed(kv)

    for token in range(1, PAGE_SIZE + 2):
        _decode(kv, "r0", 9000 + token)

    assert after_prompt == 1, (
        "a node whose decode ids are not the token it sampled keyed a page of "
        "them anyway"
    )
    assert _indexed(kv) == 1, (
        "a node whose decode ids are not the token it sampled indexed pages "
        "it cannot name"
    )
    kv.assert_pages_conserved()


def test_a_request_that_opted_out_chains_nothing():
    kv = _manager()
    prompt = list(range(PAGE_SIZE))
    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": chain(_pages(prompt))},
        prefix_tail={"main": []},
        prefix_decode={"main": TENSOR},
        kv_layout={"main": _layout(prompt)},
        loop_caps={"decode_loop": PAGE_SIZE + 1},
        prefix_cache=False,
    ))

    _prefill(kv, "r0", prompt)
    for token in range(1, PAGE_SIZE + 2):
        _decode(kv, "r0", 9000 + token)

    assert _indexed(kv) == 0, (
        "a request that turned the cache off left its generated pages behind"
    )
    kv.assert_pages_conserved()


@pytest.mark.parametrize("speculated_step_ran", [False, True])
def test_text_appended_after_a_reply_is_keyed_like_the_ids_around_it(speculated_step_ran):
    """A tool's text is prefilled after the reply it answers. Its ids are no sampled id, but the row
    brings them, and they join the chain after the reply's, so a later request whose prompt is the whole
    history finds the pages that hold them. The row begins with the reply's stop token, which the step
    speculated past the stop may have written already; the probe is handed the row whole either way (the decoder
    drops the id only when the step is declared), so it keys by the position the row says it starts at."""
    kv = KVManager(
        cfg=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=8192, max_num_pages=64,
                          page_size=PAGE_SIZE),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=torch.device("cpu"),
        dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, {"main": KeyedWalks(writes=("prefill",), growth=(WALK,))})
    prompt = list(range(100))
    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": chain(_pages(prompt))}, prefix_tail={"main": prompt},
        prefix_decode={"main": TENSOR},
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(len(prompt), len(prompt), "prefill"),), growth=(("decode_loop", 1),), keyed_spans=1)},
        loop_caps={"decode_loop": 400},
    ))
    history = list(prompt)

    def step(walk, span):
        s = KVStep(segments=(Segment("r0", "main", span),))
        ctx = StepContext(request_ids=("r0",), graph_walk=walk, slot=0, capture=False)
        assert kv.admit(s, ctx).ok
        kv.plan(s, ctx)
        kv.commit(s, ctx)

    def sample(token):
        history.append(token)
        _sampled(kv, "r0", token)

    step("prefill", len(prompt))
    sample(9000)
    for i in range(1, 141):
        step(WALK, 1)
        sample(9000 + i)
    stop = history[-1]
    if speculated_step_ran:
        step(WALK, 1)                      # writes the stop token; its own sample is dropped
    text = list(range(500, 530))
    row = [stop, *text]
    inputs = ARNodeInputs(input_seq_len=len(row), input_ids=torch.tensor(row), starts_at=len(history) - 1)
    # the scheduler measures the row first, with a row that holds only its length
    assert kv.resolve_cached_prefix("r0", NODE, "prefill", _MeasuredRow(len(row))) is None
    for _ in range(2):                     # a probe is repeated whenever the step is prepared again
        assert kv.resolve_cached_prefix("r0", NODE, "prefill", inputs) is None
    history += text
    step("prefill", len(row) - speculated_step_ran)
    sample(9500)
    for i in range(1, 31):
        step(WALK, 1)
        sample(9500 + i)
    assert kv._streams["r0"]["main"].chain is not None
    assert _indexed(kv) == 2

    later = history + list(range(700, 760))
    kv.ingest_request("r1", KVReqConfig(
        prefix_keys={"main": chain(_pages(later))}, prefix_tail={"main": later[2 * PAGE_SIZE:]},
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(len(later), len(later), "prefill"),), growth=(("decode_loop", 1),), keyed_spans=1)},
        loop_caps={"decode_loop": 0},
    ))
    got = kv.resolve_cached_prefix("r1", NODE, "prefill", ARNodeInputs(input_seq_len=len(later)))
    assert got.tokens == 2 * PAGE_SIZE, "the page holding the appended text was keyed under other ids"
    kv.assert_pages_conserved()


def _appended_kv(prompt):
    kv = KVManager(
        cfg=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=8192, max_num_pages=64,
                          page_size=PAGE_SIZE),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=torch.device("cpu"),
        dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, {"main": KeyedWalks(writes=("prefill", "stream"), growth=(WALK,))})
    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": chain(_pages(prompt))}, prefix_tail={"main": prompt},
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(len(prompt), len(prompt), "prefill"),), growth=(("decode_loop", 1),), keyed_spans=1)},
        loop_caps={"decode_loop": 400},
    ))
    s = KVStep(segments=(Segment("r0", "main", len(prompt)),))
    ctx = StepContext(request_ids=("r0",), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(s, ctx).ok
    kv.plan(s, ctx)
    kv.commit(s, ctx)
    return kv


def test_a_spoken_turns_suffix_is_keyed_after_its_transcript_so_the_chain_survives_the_replys_first_decode():
    prompt = list(range(100))
    kv = _appended_kv(prompt)
    stream = kv._streams["r0"]["main"]

    def row(walk, ids, start):
        inputs = ARNodeInputs(input_seq_len=len(ids), input_ids=torch.tensor(ids), starts_at=start)
        kv.resolve_cached_prefix("r0", NODE, walk, inputs)
        s = KVStep(segments=(Segment("r0", "main", len(ids)),))
        ctx = StepContext(request_ids=("r0",), graph_walk=walk, slot=0, capture=False)
        assert kv.admit(s, ctx).ok
        kv.plan(s, ctx)
        kv.commit(s, ctx)

    chunk, suffix = list(range(500, 509)), list(range(600, 605))
    row("stream", chunk, len(prompt))
    row("prefill", suffix, len(prompt) + len(chunk))        # the suffix: after the last chunk
    assert stream.chain.covered_len == len(prompt) + len(chunk) + len(suffix)
    row(WALK, [1], None)                                      # the reply's first decode step commits
    assert stream.chain is not None, "the chain was dropped as behind the stream: nothing of the reply is keyed"


def test_a_row_that_starts_past_the_end_of_the_chain_stops_the_chain_instead_of_misnaming_pages():
    prompt = list(range(100))
    kv = _appended_kv(prompt)
    chain_ = kv._streams["r0"]["main"].chain
    kv.resolve_cached_prefix("r0", NODE, "stream", ARNodeInputs(
        input_seq_len=3, input_ids=torch.tensor([1, 2, 3]), starts_at=len(prompt) + 5))
    assert chain_.unkeyed is None and chain_.covered_len == len(prompt)
