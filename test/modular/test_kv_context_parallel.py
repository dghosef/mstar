"""Context parallelism on a paged KV pool: each page's slots striped over the ranks that
would otherwise hold the same KV head, with logical page ids the same on every rank."""

from __future__ import annotations

import sys
from dataclasses import dataclass

sys.path.insert(0, ".")

import numpy as np
import pytest
import torch
from test_kv_prefix_correctness import NODE, PAGE_SIZE, ROOT
from test_kv_prefix_lease import _StubTransfer
from test_kv_prefix_tp import WALKS, _ctx, _follow, _ingest, _send, _step, _tokens

from mstar.distributed.communication import CommGroup, JointGroups, WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.resources.base import Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.kv.transfer import KVReadInfo, ShmKVTransferEngine
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

P = 2
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _group(rank: int, size: int) -> CommGroup:
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(size)))


def _placed(rank: int, **overrides) -> PagedKVConfig:
    cfg = dict(
        num_layers=2, num_kv_heads=1, head_dim=2, max_seq_len=64, num_qo_heads=2, max_num_pages=4,
        page_size=8, context_parallel=True,
    )
    cfg.update(overrides)
    config = PagedKVConfig(**cfg)
    config.place(JointGroups(tp_group=_group(rank, P), sp_group=CommGroup.trivial()), "kv")
    return config


def _manager(rank: int) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2, max_num_pages=12,
            page_size=PAGE_SIZE, prefix_cache_tp=True, context_parallel=True,
        ),
        name="kv", joint_comm_group=JointGroups(tp_group=_group(rank, P), sp_group=CommGroup.trivial()),
        transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, WALKS)
    kv._plan_cache_mode = 1
    kv._cg_max_seq_len = 64
    return kv


def _decode(kv: KVManager, rid: str, slot: int) -> tuple[KVStep, StepContext]:
    """One leased decode step of ``rid``, planned and not committed."""
    step = KVStep(segments=(Segment(rid, "main", 1),))
    ctx = StepContext(
        request_ids=(rid,), graph_walk="decode", slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1)),
    )
    ctx.set_padded_rids((rid,))
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    return step, ctx


def _assert_split_writes(ranks: list[KVManager], step: KVStep) -> None:
    """The ranks hold the same pages, and each token ``step`` writes goes to exactly one of
    them, the one owning its logical slot, at its physical slot; the other sends it to the sink."""
    logical = []
    for seg in step.segments:
        streams = [kv._streams[seg.request_id][seg.label] for kv in ranks]
        assert all(s.page_indices == streams[0].page_indices for s in streams), seg.request_id
        start = streams[0].stored_len
        pages = streams[0].page_indices
        logical += [(pages[pos // PAGE_SIZE], pos % PAGE_SIZE) for pos in range(start, start + seg.span)]
    writes = [list(zip(*(t.tolist() for t in kv.write_slots("main")), strict=True)) for kv in ranks]
    assert all(len(w) == len(logical) for w in writes)
    for token, (page, offset) in enumerate(logical):
        owners = [(rank, w[token]) for rank, w in enumerate(writes) if w[token] != (SINK_PAGE, 0)]
        assert owners == [(offset % P, (page, offset // P))], f"token {token}, logical ({page}, {offset})"


def test_two_ranks_keep_logical_pages_alike_and_split_every_write():
    ranks = [_manager(0), _manager(1)]
    leader, follower = ranks
    a = _tokens(3 * PAGE_SIZE + 5, seed=1)
    for kv in ranks:
        _ingest(kv, "a", a)
    steps = [_step(kv, "a", len(a)) for kv in ranks]
    _assert_split_writes(ranks, steps[0])
    for kv, step in zip(ranks, steps, strict=True):
        kv.commit(step, _ctx("a"))

    # a prefix hit rank 0 decides and rank 1 adopts
    b = a[:3 * PAGE_SIZE] + _tokens(7, seed=2)
    for kv in ranks:
        _ingest(kv, "b", b)
    assert _follow(follower, _send(leader, "b"))
    steps = [_step(kv, "b", len(b)) for kv in ranks]
    assert steps[0].segments[0].span == 7
    _assert_split_writes(ranks, steps[0])
    for kv, step in zip(ranks, steps, strict=True):
        kv.commit(step, _ctx("b"))
    assert leader._streams["b"]["main"].page_indices[:3] == leader._streams["a"]["main"].page_indices[:3]

    # decode, the second step planned off the first (the plan cache)
    for slot in (0, 1):
        planned = [_decode(kv, "b", slot) for kv in ranks]
        _assert_split_writes(ranks, planned[0][0])
        for kv, (step, ctx) in zip(ranks, planned, strict=True):
            kv.commit(step, ctx)
    assert [kv.plan_cache_stats["hit"] for kv in ranks] == [1, 1]

    # a request larger than the pool is refused alike
    for kv in ranks:
        kv.ingest_request("big", KVReqConfig())
        assert not kv.admit(KVStep(segments=(Segment("big", "main", 12 * PAGE_SIZE),)), _ctx("big")).ok
        kv.remove_request("big")

    # abort both; a new request evicts what they left cached
    c = _tokens(9 * PAGE_SIZE + 3, seed=3)
    for kv in ranks:
        kv.remove_request("a")
        kv.remove_request("b")
        _ingest(kv, "c", c)
    steps = [_step(kv, "c", len(c)) for kv in ranks]
    _assert_split_writes(ranks, steps[0])


@pytest.mark.parametrize("rank", range(P))
def test_a_placed_cache_maps_logical_token_ranges_to_this_ranks_slots(rank, tmp_path):
    source, destination = (KVCache(_placed(rank), CPU, torch.float32) for _ in range(2))
    assert source.page_slots == 8 // P and source.tensor.shape[3] == 8 // P
    source.tensor.copy_(torch.arange(1, source.tensor.numel() + 1, dtype=torch.float32).reshape(source.tensor.shape))
    # logical slots 3-5 of a page: rank 0 owns 4 (physical 2), rank 1 owns 3 and 5 (physical 1 and 2)
    first, end = {0: (2, 3), 1: (1, 3)}[rank]
    page = source.tensor[:, 2]
    assert torch.equal(source.chunk_view(1, 2, 3, 6), page[1, :, first:end])
    rows = source.layer_rows(0, 2, 2, 3, 6)
    assert [r.tolist() for r in rows] == [page[:, kv, first:end].tolist() for kv in (0, 1)]
    ptrs, nbytes = source.chunk_ptrs(1, 2, 3, 6)
    assert ptrs == [page[1, kv, first].data_ptr() for kv in (0, 1)] and nbytes == page[1, 0, first:end].nbytes

    # an SHM read of a partial page lands in this rank's slots of it, and nowhere else
    producer = ShmKVTransferEngine(source, "producer", str(tmp_path))
    consumer = ShmKVTransferEngine(destination, "consumer", str(tmp_path))
    info = producer.get_kv_transfer_info(request_id="r", label="main", page_indices=[1, 2], seq_len=14)
    assert consumer.read_batched_async(info, [KVReadInfo(layer, 0, 2, 3, 6) for layer in range(2)]) is None
    assert torch.equal(destination.tensor[:, 0, :, first:end], page[:, :, first:end])
    assert destination.tensor.count_nonzero() == page[:, :, first:end].numel()

    # the attention plan's local lengths: rows of 1 page holding 1 and 3 slots, and of 2 pages, the last full
    last, kv_lens = source.config.placement.local_last_page_len(np.array([1, 3, 8]), np.array([1, 1, 2]), 8)
    assert (last.tolist(), kv_lens.tolist()) == {0: ([1, 2, 4], [1, 2, 8]), 1: ([0, 1, 4], [0, 1, 8])}[rank]


@pytest.mark.parametrize("overrides, match", [
    (dict(page_size=5), "doesn't divide page_size 5"),
    (dict(num_kv_heads=2), "leave none sharing one"),
])
def test_a_placement_that_cannot_stripe_is_refused(overrides, match):
    with pytest.raises(ValueError, match=match):
        _placed(0, **overrides)


class _Dependent(Resource):
    supports_context_parallel = False

    @classmethod
    def build(cls, spec, info):
        resource = (_Supporting if spec.supports else cls)()
        resource.info = info
        return resource


class _Supporting(_Dependent):
    supports_context_parallel = True


@dataclass
class _DependentSpec(NodeResourceSpec):
    supports: bool = False

    @property
    def resource_class(self):
        return _Dependent

    def depends_on(self):
        return {"kv"}


class _Submodule:
    disable_autocast = True
    forward = forward_batched = None

    def requires_grad_(self, flag):
        return self

    def bind_node_resources(self, resources):
        self.resources = resources


@pytest.mark.parametrize("supports", [False, True])
def test_an_sp_placed_node_shards_heads_over_tp_and_refuses_a_dependent_without_support(supports):
    groups = WorkerParallelGroups(num_workers=P, global_rank=0)
    groups.add_sp(NODE, _group(0, P))
    kv_spec = KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=256, num_qo_heads=4, max_num_pages=8,
        page_size=PAGE_SIZE, context_parallel=True,
    ))
    specs = [kv_spec, _DependentSpec(resource_key="attn", nodes={NODE}, supports=supports)]
    engine = Engine(graph_runtime=None)
    load = lambda: engine.load_model(  # noqa: E731
        {NODE: _Submodule()}, specs=specs, parallel_groups=groups, device=CPU, transfer_engine_info=None,
    )
    if not supports:
        with pytest.raises(ValueError, match="'attn'.*does not support context parallelism.*'kv'"):
            load()
        return
    load()
    # heads over TP alone (1 rank), positions over the 2 SP ranks
    assert engine._resources["attn"].info.head_shards == 1 and engine._submodules[NODE].head_shards == 1
    assert (kv_spec.config.num_kv_heads, kv_spec.config.num_qo_heads) == (2, 4)
    placement = kv_spec.config.placement
    assert (placement.size, placement.index, placement.over_sp) == (P, 0, True)
    assert engine._resources["kv"].kv_cache.page_slots == PAGE_SIZE // P
