"""Context parallelism on a paged KV pool: each page's slots striped over the ranks that
would otherwise hold the same KV head, with logical page ids the same on every rank; and
attention over it, each rank's partial over its slots exchanged and merged."""

from __future__ import annotations

import itertools
import math
import os
import sys
import types
from dataclasses import dataclass, replace

sys.path.insert(0, ".")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import Version
from test_kv_prefix_correctness import NODE, PAGE_SIZE, ROOT
from test_kv_prefix_lease import _StubTransfer
from test_kv_prefix_tp import WALKS, _ctx, _follow, _ingest, _send, _step, _tokens
from test_startup_graph_validation import _qwen3_5

from mstar.distributed.communication import CommGroup, JointGroups, WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.resources.attn import flashinfer_mla as fm
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep, AttnBackend
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVLayout, KVReqConfig, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutput, KVPlanOutputs, SequenceView, build_paged_indptrs
from mstar.engine.resources.kv.transfer import KVReadInfo, ShmKVTransferEngine
from mstar.engine.resources.spec import NodeResourceSpec, apply_yaml_overrides
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
from mstar.model.qwen3_5.config import GDN_STATE, KV_CACHE

P = 2
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _group(rank: int, size: int) -> CommGroup:
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(size)))


def _placed(rank: int) -> KVCache:
    config = PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=2, max_seq_len=64, num_qo_heads=2, max_num_pages=4,
        page_size=8, context_parallel=True,
    )
    joint = JointGroups(tp_group=_group(rank, P), sp_group=CommGroup.trivial())
    return KVCache(config, CPU, torch.float32, config.place(joint, "kv"))


def _manager(rank: int, cfg: PagedKVConfig) -> KVManager:
    # what `KVManager.build` does first: TP placement shards heads over the TP ranks
    cfg.shard(P)
    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=JointGroups(tp_group=_group(rank, P), sp_group=CommGroup.trivial()),
        transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, WALKS)
    kv._plan_cache_mode = 1
    kv._cg_max_seq_len = 64
    return kv


def _decode(kv: KVManager, rid: str, slot: int, leased: bool = True) -> tuple[KVStep, StepContext]:
    """One decode step of ``rid``, planned and not committed: into a capture slot's buffers when
    ``leased``, else eager."""
    step = KVStep(segments=(Segment(rid, "main", 1),))
    lease = SlotLease(slot=slot, bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1)) if leased else None
    ctx = StepContext(request_ids=(rid,), graph_walk="decode", slot=slot, capture=False, slot_lease=lease)
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
    # one config for both, as one process emulating TP builds them: each rank's placement is its own
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2, max_num_pages=12,
        page_size=PAGE_SIZE, prefix_cache_tp=True, context_parallel=True,
    )
    ranks = [_manager(0, cfg), _manager(1, cfg)]
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
    planned = [_decode(kv, "b", 0, leased=False) for kv in ranks]
    _assert_split_writes(ranks, planned[0][0])
    for kv, (step, ctx) in zip(ranks, planned, strict=True):
        kv.commit(step, ctx)

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
    source, destination = (_placed(rank) for _ in range(2))
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
    # a range holding none of this rank's slots is no transfer at all
    other = (1 - rank, 2 - rank)
    assert source.layer_rows(0, 2, 2, *other) == [] and source.chunk_ptrs(1, 2, *other) == ([], 0)

    # an SHM read of a partial page lands in this rank's slots of it, and nowhere else
    producer = ShmKVTransferEngine(source, "producer", str(tmp_path))
    consumer = ShmKVTransferEngine(destination, "consumer", str(tmp_path))
    info = producer.get_kv_transfer_info(request_id="r", label="main", page_indices=[1, 2], seq_len=14)
    assert consumer.read_batched_async(info, [KVReadInfo(layer, 0, 2, 3, 6) for layer in range(2)]) is None
    assert torch.equal(destination.tensor[:, 0, :, first:end], page[:, :, first:end])
    assert destination.tensor.count_nonzero() == page[:, :, first:end].numel()


class _Dependent(Resource):
    @classmethod
    def build(cls, spec, info):
        resource = cls()
        resource.supports_context_parallel = spec.supports
        resource.info = info
        return resource


@dataclass
class _DependentSpec(NodeResourceSpec):
    supports: bool = False

    @property
    def resource_class(self):
        return _Dependent

    def depends_on(self):
        return {KV_CACHE}


class _Submodule:
    disable_autocast = True
    forward = forward_batched = None

    def requires_grad_(self, flag):
        return self

    def bind_node_resources(self, resources):
        self.resources = resources


@pytest.mark.parametrize("supports", [False, True])
def test_an_sp_placed_node_shards_heads_over_tp_and_refuses_a_dependent_without_support(
    supports, tmp_path, monkeypatch,
):
    groups = WorkerParallelGroups(num_workers=P, global_rank=0)
    groups.add_sp(NODE, _group(0, P))
    # Qwen3.5's own declarations (dummy shapes) with its pool placed, as on SP ranks; then an MLA
    # pool with its attention (GLM-5.x without DSA), and a dependent without support
    specs = _qwen3_5(NODE, {}, tmp_path, monkeypatch).get_node_resources()
    apply_yaml_overrides(specs, {"resources": {
        KV_CACHE: {"context_parallel": True, "max_num_pages": 8}, GDN_STATE: {"max_slots": 4},
    }})
    kv_spec, = (spec for spec in specs if spec.resource_key == KV_CACHE)
    heads = (kv_spec.config.num_kv_heads, kv_spec.config.num_qo_heads)
    specs += [
        KVSpec(resource_key="mla_kv", nodes={NODE}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=LATENT, max_seq_len=256, num_qo_heads=4, max_num_pages=8,
            page_size=PAGE_SIZE, context_parallel=True, layout=KVLayout.MLA, kv_lora_rank=CKV, qk_rope_head_dim=KPE,
        )),
        AttentionSpec(resource_key="mla_attn", nodes={NODE}, config=AttentionConfig(
            kv_cache="mla_kv", backend=AttnBackend.FLASHINFER_MLA, sm_scale=LATENT ** -0.5,
        )),
        _DependentSpec(resource_key="dependent", nodes={NODE}, supports=supports),
    ]
    engine = Engine(graph_runtime=None)
    load = lambda: engine.load_model(  # noqa: E731
        {NODE: _Submodule()}, specs=specs, parallel_groups=groups, device=CPU, transfer_engine_info=None,
    )
    if not supports:
        with pytest.raises(ValueError, match=f"'dependent'.*does not support context parallelism.*'{KV_CACHE}'"):
            load()
        return
    load()
    # heads over TP alone (1 rank), positions over the 2 SP ranks
    assert engine._resources["dependent"].info.head_shards == 1 and engine._submodules[NODE].head_shards == 1
    assert (kv_spec.config.num_kv_heads, kv_spec.config.num_qo_heads) == heads
    placement = engine._resources[KV_CACHE].placement
    assert (placement.size, placement.index, placement.over_sp) == (P, 0, True)
    assert engine._resources[KV_CACHE].kv_cache.page_slots == kv_spec.config.page_size // P
    # built by hand, as a harness does, with a node's info told nothing about the placement: a
    # resource over the sharded pool that shards it another way is refused
    unplaced = EngineResourceInfo(device=CPU, joint_comm_group=groups.get_joint_group_for_node(NODE),
                                  nodes=frozenset({NODE}))
    with pytest.raises(ValueError, match="already sharded 1 ways.*2 ways"):
        AttentionManager.build(
            AttentionSpec(resource_key="attn2", nodes={NODE}, config=AttentionConfig(kv_cache=KV_CACHE)),
            replace(unplaced, dependencies={KV_CACHE: kv_spec}),
        )


# --- attention over a placed pool -------------------------------------------------------------

HEAD_DIM, Q_HEADS, CP_PAGE = 64, 4, 4
# MLA: the latent [ckv | kpe], in the one shape FlashInfer's MLA kernel takes
CKV, KPE = 512, 64
LATENT = CKV + KPE
# (stored context, new tokens) per row. Prefill: a row with no context, one whose 1-token
# context only rank 0 holds, and contexts ending on a page whose one token rank 1 lacks.
PREFILL_ROWS = ((0, 5), (1, 3), (5, 2), (6, 1))
# decode, after the write: lengths 1 (rank 1 holds nothing), 5 (a last page rank 1 lacks), 7
DECODE_ROWS = ((0, 1), (4, 1), (6, 1))


def _reference(q, k, v, causal=True):
    """Attention of ``q`` ``[T, Hq, D]`` over ``k``/``v`` ``[L, Hkv, D]``, the queries the last ``T``
    positions when causal; fp32 ``(out, lse)``, lse natural-log ``[T, Hq]``."""
    group = q.shape[1] // k.shape[1]
    k, v = (t.float().repeat_interleave(group, dim=1) for t in (k, v))
    scores = torch.einsum("thd,lhd->htl", q.float(), k) * q.shape[-1] ** -0.5
    if causal:
        t, length = q.shape[0], k.shape[0]
        seen = torch.arange(length)[None, :] <= torch.arange(t)[:, None] + (length - t)
        scores = scores.masked_fill(~seen.to(scores.device), -math.inf)
    return torch.einsum("htl,lhd->thd", torch.softmax(scores, dim=-1), v), torch.logsumexp(scores, -1).T


def _reference_flashinfer() -> types.ModuleType:
    """FlashInfer's paged prefill/decode and ragged wrappers as dense torch math on CPU, with
    their lse bases. A paged plan refuses a row FlashInfer's split-KV planner would miscount: no
    pages, or a last page holding no slot (see ``ContextParallel.local_indptrs``)."""
    fi = types.ModuleType("flashinfer")

    class Paged:
        base = "log2"

        def __init__(self, *args, **kwargs):
            pass

        def plan(self, qo_indptr=None, paged_kv_indptr=None, paged_kv_indices=None,
                 paged_kv_last_page_len=None, *, page_size, seq_lens, causal=False, indptr=None,
                 indices=None, last_page_len=None, **kwargs):
            kvp = (paged_kv_indptr if indptr is None else indptr).tolist()
            self.indices = (paged_kv_indices if indices is None else indices).tolist()
            last = (paged_kv_last_page_len if last_page_len is None else last_page_len).tolist()
            self.rows = [(kvp[b], kvp[b + 1], int(seq_lens[b])) for b in range(len(last))]
            for (start, end, length), tail in zip(self.rows, last, strict=True):
                assert end > start and 1 <= tail <= page_size and length == (end - start - 1) * page_size + tail
            self.qo = qo_indptr.tolist() if qo_indptr is not None else list(range(len(last) + 1))
            self.causal = causal

        def run(self, q, kv, return_lse=False, lse_base=None):
            outs, lses = [], []
            for b, (start, end, length) in enumerate(self.rows):
                pages = kv[self.indices[start:end]]
                k, v = (pages[:, i].flatten(0, 1)[:length] for i in (0, 1))
                out, lse = _reference(q[self.qo[b]:self.qo[b + 1]], k, v, self.causal)
                outs.append(out)
                lses.append(lse)
            out, lse = torch.cat(outs).to(q.dtype), torch.cat(lses)
            if (lse_base or self.base) == "log2":
                lse = lse / math.log(2)
            return (out, lse) if return_lse else out

    class Decode(Paged):
        def __init__(self, *args, use_tensor_cores=False, **kwargs):
            self.use_tensor_cores = use_tensor_cores

        def plan(self, **kwargs):
            super().plan(**kwargs)
            self.causal = False

    class Ragged:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self, qo_indptr, kv_indptr, *args, causal=False, **kwargs):
            self.qo, self.causal = qo_indptr.tolist(), causal

        def run(self, q, k, v, out=None, return_lse=False, lse_base="log2"):
            parts = [_reference(q[a:b], k[a:b], v[a:b], self.causal) for a, b in itertools.pairwise(self.qo)]
            o, lse = torch.cat([p[0] for p in parts]).to(q.dtype), torch.cat([p[1] for p in parts])
            lse = lse if lse_base == "ln" else lse / math.log(2)
            return (o, lse) if return_lse else o

    fi.BatchPrefillWithPagedKVCacheWrapper = Paged
    fi.BatchDecodeWithPagedKVCacheWrapper = Decode
    fi.BatchPrefillWithRaggedKVCacheWrapper = Ragged
    return fi


@dataclass
class _Case:
    """One step's KV plan, the logical keys of its rows, and its queries. Under MLA a key is the
    latent, which is also its value (the output being the ``ckv`` part)."""
    plan: KVPlanOutputs
    keys: list[torch.Tensor]       # per row, every position's [2, 1, D] K and V, or [LATENT]
    q: torch.Tensor                # [N, Q_HEADS, D or LATENT]
    rows: tuple[tuple[int, int], ...]
    mla: bool

    @classmethod
    def build(cls, rows, device, dtype, mla=False) -> _Case:
        gen = torch.Generator().manual_seed(0)
        views, keys, page = [], [], 1
        key_shape, width = ((LATENT,), LATENT) if mla else ((2, 1, HEAD_DIM), HEAD_DIM)
        for i, (context, new) in enumerate(rows):
            length = context + new
            pages = list(range(page, page + -(-length // CP_PAGE)))
            page += len(pages)
            views.append(SequenceView(f"r{i}", "main", pages, length, new))
            keys.append(torch.randn(length, *key_shape, generator=gen).to(device, dtype))
        q = torch.randn(sum(new for _, new in rows), Q_HEADS, width, generator=gen).to(device, dtype)
        indptrs = build_paged_indptrs(views, CP_PAGE)
        return cls(KVPlanOutputs({"main": KVPlanOutput(indptrs, views)}), keys, q, rows, mla)

    def cache(self, placement) -> torch.Tensor:
        """This rank's layer of the pool: each owned slot at its physical place, NaN elsewhere."""
        slots = CP_PAGE // placement.size
        pages = max(view.page_idxs[-1] for view in self.plan["main"].views) + 1
        shape = (pages, slots, LATENT) if self.mla else (pages, 2, slots, 1, HEAD_DIM)
        cache = torch.full(shape, math.nan, device=self.q.device, dtype=self.q.dtype)
        for view, kv in zip(self.plan["main"].views, self.keys, strict=True):
            for pos in range(view.length):
                offset = pos % CP_PAGE
                if offset % placement.size == placement.index:
                    page, slot = view.page_idxs[pos // CP_PAGE], offset // placement.size
                    (cache[page, slot] if self.mla else cache[page, :, slot]).copy_(kv[pos])
        return cache

    def queries(self, heads=slice(None)) -> tuple[torch.Tensor, ...]:
        """The query ``run`` takes for ``heads``: ``(q,)``, or under MLA ``(q_nope, q_pe)``."""
        q = self.q[:, heads]
        return (q[..., :CKV].contiguous(), q[..., CKV:].contiguous()) if self.mla else (q.contiguous(),)

    def new_keys(self) -> tuple[torch.Tensor, ...]:
        """The step's own tokens' keys as the layer passes them to ``run``: K and V ``[N, 1, D]``,
        or under MLA the latents ``[N, LATENT]``."""
        new = torch.cat([kv[context:] for kv, (context, _) in zip(self.keys, self.rows, strict=True)])
        return (new,) if self.mla else (new[:, 0], new[:, 1])

    def runner(self, attn, placement, heads=slice(None), keys=True):
        """``attn.run`` for ``heads`` over this rank's cache, passed the new keys when ``keys``, as a
        call whose inputs are built once (so it can be captured)."""
        q, cache = self.queries(heads), self.cache(placement)
        new = self.new_keys() if keys else (None, None)
        if self.mla:
            return lambda: attn.run(q[0], "main", kv_cache_layer=cache, k=new[0], q_pe=q[1])
        return lambda: attn.run(q[0], "main", kv_cache_layer=cache, k=new[0], v=new[1])

    def want(self) -> torch.Tensor:
        outs = []
        qo = 0
        for kv, (_, new) in zip(self.keys, self.rows, strict=True):
            k, v = (kv[:, None], kv[:, None]) if self.mla else (kv[:, 0], kv[:, 1])
            out = _reference(self.q[qo:qo + new], k, v)[0]
            outs.append(out[..., :CKV] if self.mla else out)
            qo += new
        return torch.cat(outs)


def _cp_attention(
    joint: JointGroups, device: torch.device, dtype: torch.dtype, mla=False, num_kv_heads=1, num_qo_heads=Q_HEADS,
    **attention,
):
    """A ``FlashInferManager`` (``FlashInferMLAManager`` for ``mla``) over a placed pool, and this
    rank's placement."""
    layout = dict(head_dim=HEAD_DIM)
    if mla:
        layout = dict(head_dim=LATENT, layout=KVLayout.MLA, kv_lora_rank=CKV, qk_rope_head_dim=KPE)
        attention.update(backend=AttnBackend.FLASHINFER_MLA, sm_scale=LATENT ** -0.5)
    kv = KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=num_kv_heads, max_seq_len=64, num_qo_heads=num_qo_heads,
        max_num_pages=16, page_size=CP_PAGE, context_parallel=True, **layout,
    ))
    info = EngineResourceInfo(
        device=device, joint_comm_group=joint, kv_dtype=dtype, dependencies={"kv": kv},
        positions_over_sp=joint.sp_group.world_size > 1,
    )
    spec = AttentionSpec(resource_key="attn", nodes={NODE}, config=AttentionConfig(kv_cache="kv", **attention))
    attn = AttentionManager.build(spec, info)
    # the KV resource's one field attention reads
    placement = kv.config.place(joint, "kv")
    attn.bind_dependencies({"kv": types.SimpleNamespace(placement=placement)})
    return attn, placement


def _plan(attn, case: _Case, slot: int = 0, bucket: BucketKey | None = None, **step) -> None:
    ctx = StepContext(request_ids=tuple(v.request_id for v in case.plan["main"].views), graph_walk="w",
                      slot=slot, capture=False, slot_lease=None if bucket is None else SlotLease(slot, bucket))
    ctx.plan_results["kv"] = case.plan
    attn.plan(AttentionStep(**step), ctx)


def _over_ranks(rank, world, backend, store, over_sp, rows, mla, captured=False):
    """One rank of a ``world``-rank placement running ``rows`` through ``FlashInferManager.run``
    against one rank attending every key; every rank issues the same collectives, same shapes.
    ``captured`` (NCCL only) plans the step for a graph bucket, and captures ``run`` with its
    collectives in a CUDA graph on every rank and replays it."""
    os.environ["MSTAR_WORKSPACE_BUFFER_MB"] = "1"
    dist.init_process_group(backend, init_method=f"file://{store}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    cuda = backend == "nccl"
    device = torch.device("cuda", rank) if cuda else CPU
    if cuda:
        torch.cuda.set_device(device)
    else:
        sys.modules["flashinfer"] = _reference_flashinfer()
    calls = []
    for name in ("all_to_all_single", "all_gather_into_tensor"):
        def record(*args, _name=name, _call=getattr(dist, name), **kwargs):
            calls.append((_name, tuple(args[0].shape), tuple(args[1].shape), *(str(a) for a in args[2:])))
            return _call(*args, **kwargs)
        setattr(dist, name, record)
    try:
        group = CommGroup(rank, rank, list(range(world)))
        group.device_group, group.initialized = dist.group.WORLD, True
        joint = (JointGroups(tp_group=CommGroup.trivial(), sp_group=group) if over_sp
                 else JointGroups(tp_group=group, sp_group=CommGroup.trivial()))
        dtype = torch.bfloat16 if cuda else torch.float32
        attn, placement = _cp_attention(joint, device, dtype, mla=mla)
        case = _Case.build(rows, device, dtype, mla)
        # the slot plans the other kind of step first
        _plan(attn, _Case.build(DECODE_ROWS if rows == PREFILL_ROWS else PREFILL_ROWS, device, dtype, mla))
        _plan(attn, case, bucket=BucketKey("w", bs=len(rows), num_tokens=case.q.shape[0]) if captured else None)
        # a one-token-a-row step attends the pages alone, so a fused KV-write kernel can serve it
        assert attn.reads_step_kv() == (rows == PREFILL_ROWS)
        heads = slice(None) if over_sp else slice(rank * Q_HEADS // world, (rank + 1) * Q_HEADS // world)
        if rows == PREFILL_ROWS:
            with pytest.raises(ValueError, match="got none"):
                case.runner(attn, placement, heads, keys=False)()
        tol = {"atol": 2e-2, "rtol": 2e-2} if cuda else {}
        run = case.runner(attn, placement, heads)
        out = run()
        if captured:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            # what the check reads is the replay's
            out.fill_(math.nan)
            graph.replay()
            torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), case.want()[:, heads], **tol)
        if rows == PREFILL_ROWS and not captured:
            # an eager first prefill (no row holds stored context) attends its own tokens alone, on
            # every rank, with no collective
            first = _Case.build(((0, 5), (0, 3)), device, dtype, mla)
            _plan(attn, first)
            issued = len(calls)
            torch.testing.assert_close(first.runner(attn, placement, heads)().float(), first.want()[:, heads], **tol)
            assert len(calls) == issued
        everyone = [None] * world
        dist.all_gather_object(everyone, calls)
        assert calls and all(c == calls for c in everyone), everyone
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("attention, refused", [
    (dict(sliding_window=8), "sliding_window"),
    # on a host without FA3 dense degrades to FlashInfer, which would serve the pool
    (dict(backend=AttnBackend.DENSE), "backend 'dense'"),
    # its decode lse base isn't known
    (dict(flashinfer_backend="cudnn"), "flashinfer_backend 'cudnn'"),
])
def test_attention_a_placed_pool_cannot_serve_is_refused_at_build(attention, refused):
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial())
    with pytest.raises(ValueError, match=f"'attn': {refused} is unsupported over KV pool 'kv'"):
        _cp_attention(joint, CPU, torch.float32, **attention)


# the MLA case on CPU runs the MLA wrapper's own torch fallback, which these rows check the same way
LAYOUTS = pytest.mark.parametrize("mla", [False, True], ids=["gqa", "mla"])


def _flashinfer(mla: bool) -> None:
    """Skips without FlashInfer, and for MLA on a FlashInfer a placed pool refuses at bind."""
    flashinfer = pytest.importorskip("flashinfer")
    if mla and Version(flashinfer.__version__).public != fm._FA_PLANNER_VERSION:
        pytest.skip(f"attention over a placed MLA pool refuses FlashInfer {flashinfer.__version__} at bind")


@LAYOUTS
@pytest.mark.parametrize("rows", [PREFILL_ROWS, DECODE_ROWS], ids=["prefill", "decode"])
@pytest.mark.parametrize("over_sp", [False, True], ids=["tp", "sp"])
def test_two_ranks_attend_a_placed_pool_like_one_rank_with_the_same_collectives(tmp_path, over_sp, rows, mla):
    mp.spawn(_over_ranks, args=(P, "gloo", str(tmp_path / "store"), over_sp, rows, mla), nprocs=P)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@LAYOUTS
@pytest.mark.parametrize("captured", [False, True], ids=["eager", "captured"])
@pytest.mark.parametrize("rows", [PREFILL_ROWS, DECODE_ROWS], ids=["prefill", "decode"])
@pytest.mark.parametrize("over_sp", [False, True], ids=["tp", "sp"])
def test_two_gpus_attend_a_placed_pool_like_one(tmp_path, over_sp, rows, captured, mla):
    _flashinfer(mla)
    mp.spawn(_over_ranks, args=(P, "nccl", str(tmp_path / "store"), over_sp, rows, mla, captured), nprocs=P)


def test_tp_placed_decode_runs_on_tensor_cores_whatever_decode_tensor_cores_says(monkeypatch):
    """A rank attends every partner's query heads over one KV head, a GQA group FlashInfer's
    CUDA-core decode has no kernel for unless it is one of a few sizes."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", _reference_flashinfer())
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial())
    attn, _ = _cp_attention(joint, CPU, torch.float32, decode_tensor_cores=False)
    _plan(attn, _Case.build(DECODE_ROWS, CPU, torch.float32))
    assert attn._current_plan_states["main"].context.attn_wrapper.use_tensor_cores


def test_mla_attention_refuses_a_context_only_step_over_a_placed_pool():
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial())
    attn, _ = _cp_attention(joint, CPU, torch.float32, mla=True)
    with pytest.raises(ValueError, match="context_only step over a context-parallel KV pool"):
        _plan(attn, _Case.build(PREFILL_ROWS, CPU, torch.float32, mla=True), context_only=True)


def _fa_kernel(monkeypatch, version: str = fm._FA_PLANNER_VERSION) -> None:
    """The MLA manager plans as if FlashInfer ``version``'s FA kernel served it; on CPU its
    wrappers still plan the torch fallback."""
    monkeypatch.setattr(fm, "_sm_count", lambda device: 132)
    monkeypatch.setattr(fm.FlashInferMLAManager, "uses_kernel", property(lambda self: True))
    monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace(__version__=version))


def test_mla_ranks_refuse_the_same_steps_past_the_fa_planners_work_item_cap(monkeypatch):
    """FlashInfer's FA MLA planner counts each rank's own KV lengths, which differ over a placed
    pool, so its own refusal could come on one rank alone and hang the others in the exchange.
    The refusal before it is the same on every rank, whatever the cap."""
    _fa_kernel(monkeypatch)
    case = _Case.build(PREFILL_ROWS, CPU, torch.float32, mla=True)
    ranks = [
        _cp_attention(JointGroups(tp_group=_group(rank, P), sp_group=CommGroup.trivial()), CPU, torch.float32,
                      mla=True, flashinfer_backend="fa2")
        for rank in range(P)
    ]
    # the context lengths each rank plans
    indptrs = case.plan["main"].cpu_indptrs
    lens = [place.local_indptrs(indptrs.context_only(CP_PAGE), CP_PAGE)[0].kv_lens.tolist() for _, place in ranks]
    assert lens[0] != lens[1]

    def refuses(attn) -> bool:
        try:
            _plan(attn, case)
        except ValueError as error:
            assert "work items" in str(error)
            return True
        return False

    decisions = []
    for cap in range(1, 400):
        monkeypatch.setattr(fm, "_FA_MAX_WORK_ITEMS", cap)
        decisions.append([refuses(attn) for attn, _ in ranks])
    assert all(a == b for a, b in decisions) and {a for a, _ in decisions} == {False, True}
    # the torch fallback runs no FA planner, so nothing is refused for it
    monkeypatch.setattr(fm, "_FA_MAX_WORK_ITEMS", 1)
    monkeypatch.setattr(fm.FlashInferMLAManager, "uses_kernel", property(lambda self: False))
    _plan(_cp_attention(JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial()), CPU, torch.float32,
                        mla=True, flashinfer_backend="fa2")[0], case)


@pytest.mark.parametrize("version, heads, refused", [
    ("0.7.3", Q_HEADS, r"FlashInfer 0\.7\.3 is installed"),
    # a local build of the release
    (f"{fm._FA_PLANNER_VERSION}+cu130", Q_HEADS, None),
    # TP 2 shards the heads: 20 a rank
    (fm._FA_PLANNER_VERSION, 40, "causal tile-bound check can refuse a step at 20 query heads"),
    (fm._FA_PLANNER_VERSION, 128, None),
])
def test_mla_attention_over_a_placed_pool_refuses_at_bind_a_flashinfer_or_head_count_its_fa_planner_may_refuse(
    monkeypatch, version, heads, refused,
):
    _fa_kernel(monkeypatch, version)
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial())
    if refused is None:
        _cp_attention(joint, CPU, torch.float32, mla=True, num_qo_heads=heads, flashinfer_backend="fa2")
        return
    with pytest.raises(ValueError, match=refused):
        _cp_attention(joint, CPU, torch.float32, mla=True, num_qo_heads=heads, flashinfer_backend="fa2")


def test_the_fa_work_item_bound_holds_flashinfers_own_count():
    """The refusal above is rank-symmetric only while ``fa_work_items_bound`` is at least what the
    installed FlashInfer's planner counts on any rank's KV lengths."""
    _flashinfer(mla=True)
    from flashinfer.mla._batch_mla._backends import _fa_common as fa

    assert fm._FA_MAX_WORK_ITEMS == fa._FA_MAX_WORK_ITEMS
    gen = torch.Generator().manual_seed(0)
    for _ in range(3000):
        rows = int(torch.randint(1, 6, (), generator=gen))
        q = torch.randint(1, 300, (rows,), generator=gen)
        kv = q + torch.randint(0, 20000, (rows,), generator=gen)
        heads = (8, 16, 20, 40, 64, 128)[int(torch.randint(6, (), generator=gen))]
        sm = (78, 132)[int(torch.randint(2, (), generator=gen))]
        bound = fm.fa_work_items_bound(torch.cat([torch.zeros(1, dtype=q.dtype), q.cumsum(0)]), heads, sm)
        for causal in (False, True):
            try:
                count = fa._fa_plan_work_count(q.tolist(), kv.tolist(), heads, sm, causal)
            except fa._BackendPlanUnsupportedError:
                # its query tiles alone are past the cap
                assert bound > fa._FA_MAX_WORK_ITEMS
                continue
            assert count <= bound, (q.tolist(), kv.tolist(), heads, sm, causal)
