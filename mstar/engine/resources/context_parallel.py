"""Context parallelism: a paged KV pool's token positions placed over a group of ranks
that would otherwise hold the same KV heads.

Page ids stay logical and are the same on every rank, so allocation, prefix keys,
offload and admission need nothing new. Within a page, logical slot ``s`` belongs to
partner ``s % P`` and sits at its physical slot ``s // P``, so each rank stores
``page_size // P`` slots a page. ``PagedKVConfig.place`` resolves the placement.

Attention over such a pool runs on each rank over its own slots and returns the partial with
its log-sum-exp; the partners exchange partials and ``merge`` them, exactly.
"""

import math
from collections.abc import Callable, Iterable
from typing import NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens, PagedIndptrs
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.utils.h2d import PinnedStager


class ContextParallel(NamedTuple):
    """This rank's place in its KV pool's placement.

    Under SP placement the partners are the whole SP group. Under TP placement they are a block of
    the TP group (the ranks sharing one KV head), and the exchange is an all-to-all over the whole
    TP group with nothing sent to the ranks outside the block, so no group of its own is needed."""

    # P: how many partners each page's slots stripe over
    size: int
    # r: this rank owns the logical slots s with s % P == r
    index: int
    # what the partners exchange over: the SP group in SP mode, the TP group in TP mode
    group: CommGroup
    over_sp: bool
    # the partners' ranks in ``group``, in partner-index order
    partners: tuple[int, ...]

    def slot_range(self, start: int, end: int) -> tuple[int, int]:
        """The physical slots of a page holding this rank's share of logical slots ``[start, end)``."""
        p, r = self.size, self.index
        return (start - r + p - 1) // p, (end - r + p - 1) // p

    def write_slots(self, pages, offsets):
        """Each token's write address on this rank, from its logical ``(page, offset)``: an owned
        token goes to ``(page, offset // P)``, the rest to ``(SINK_PAGE, 0)``. Device tensors stay
        on the device; anything else comes back as numpy arrays."""
        if isinstance(pages, torch.Tensor):
            where = torch.where
        else:
            where, pages, offsets = np.where, np.asarray(pages), np.asarray(offsets)
        owned = offsets % self.size == self.index
        return where(owned, pages, SINK_PAGE), where(owned, offsets // self.size, 0)

    def local_geometry(self, page_size: int, num_qo_heads: int) -> dict[str, int]:
        """A paged attention wrapper's ``page_size`` and ``num_qo_heads`` over this rank's slots: the
        slots a page it stores, and under TP placement every partner's query heads, which it attends
        for them (``exchange_queries``)."""
        return dict(page_size=page_size // self.size, num_qo_heads=num_qo_heads * (1 if self.over_sp else self.size))

    def local_last_page_len(
        self, logical_last: np.ndarray, npages: np.ndarray, page_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per row, this rank's slots in the last page (0 to ``page_size // P``) and its KV
        length, from the logical last-page lengths and page counts. The page indices and
        indptr stay as they are."""
        slots = page_size // self.size
        last = np.clip((logical_last - self.index + self.size - 1) // self.size, 0, slots)
        return last, np.where(npages > 0, (npages - 1) * slots + last, 0)

    def local_indptrs(self, indptrs: PagedIndptrs, page_size: int) -> tuple[PagedIndptrs, np.ndarray]:
        """A plan's logical paged indptrs as this rank's (``page_size`` the logical one), and per row
        whether it holds any of this rank's slots.

        Every row keeps a length of at least 1 and a last page holding at least one slot, as on one
        rank: FlashInfer's split-KV planner counts a row's pages where its kernel counts its tokens,
        and a last page with none of this rank's slots puts the two out of step. Such a page is
        dropped from its row. A row with no slot here attends one slot of SINK_PAGE instead, and its
        partial is masked before the exchange (``exchange_partials``)."""
        kvp = indptrs.paged_kv_indptr.numpy().astype(np.int64)
        npages = np.diff(kvp)
        last, lens = self.local_last_page_len(indptrs.paged_kv_last_page_len.numpy(), npages, page_size)
        valid = lens > 0
        drop = valid & (last == 0)
        kv_indptr, indices = indptrs.paged_kv_indptr, indptrs.paged_kv_indices
        if drop.any() or not valid.all():
            keep_pages = np.where(valid, npages - drop, 0)
            new_kvp = np.zeros(len(npages) + 1, dtype=np.int64)
            np.cumsum(np.where(valid, keep_pages, 1), out=new_kvp[1:])
            row = np.repeat(np.arange(len(npages)), npages)
            within = np.arange(len(row)) - kvp[row]
            keep = within < keep_pages[row]
            out = np.full(int(new_kvp[-1]), SINK_PAGE, dtype=np.int32)
            out[new_kvp[row[keep]] + within[keep]] = indices.numpy()[:len(row)][keep]
            last = np.where(valid, np.where(drop, page_size // self.size, last), 1)
            lens = np.where(valid, lens, 1)
            kv_indptr, indices = torch.from_numpy(new_kvp.astype(np.int32)), torch.from_numpy(out)
        return PagedIndptrs(
            qo_indptr=indptrs.qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=torch.from_numpy(last.astype(np.int32)),
            kv_lens=torch.from_numpy(lens.astype(np.int32)).as_subclass(HostLens),
        ), valid

    def attend(
        self,
        queries: tuple[torch.Tensor, ...],
        partial: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        new: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Attention over the whole pool for this rank's query, fp32 (``new``'s out when ``partial`` is
        None): ``queries`` its parts, each
        ``[N, H, *]`` (one, or a backend's several, such as MLA's ``q_nope`` and ``q_pe``).
        ``partial`` attends this rank's slots for the parts ``exchange_queries`` returns and gives
        ``(out, lse, valid)`` as ``exchange_partials`` takes them; ``new`` is the step's own
        tokens' ``(out, lse)`` for this rank's heads, when it attends them apart from the pool.
        Runs the same collectives on every rank; none when ``partial`` is None, which every rank
        must agree on: no row has stored context, so ``new`` is the whole attention."""
        if partial is None:
            return new[0]
        outs, lses = self.exchange_partials(*partial(*self.exchange_queries(queries)))
        if new is not None:
            outs = torch.cat([outs, new[0].to(outs.dtype).unsqueeze(0)])
            lses = torch.cat([lses, new[1].float().unsqueeze(0)])
        return merge(outs, lses)

    def exchange_queries(self, queries: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """The query parts this rank attends its slots for, from its own (each ``[N, H, *]``). SP
        partners hold the same heads, so they are its own; TP partners hold different query heads
        of one KV head, so each gets every partner's, ``[N, P * H, *]`` partner-major, the parts
        sent as one tensor."""
        if self.over_sp:
            return queries
        q = queries[0] if len(queries) == 1 else torch.cat(queries, dim=-1)
        blocks = q.unsqueeze(0).expand(self.size, *q.shape)
        q = self.group.to_partners(blocks, self.partners).transpose(0, 1).flatten(1, 2)
        return q.split([part.shape[-1] for part in queries], dim=-1)

    def exchange_partials(
        self, out: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every partner's partial for this rank's heads, ``[P, N, H, D]`` in ``out``'s dtype and
        ``[P, N, H]`` fp32, to ``merge``; from this rank's ``out`` ``[N, H', D]`` and natural-log
        ``lse`` ``[N, H']`` over its slots, for the heads ``exchange_queries`` returned. A query row
        ``valid`` (``[N]``) marks as holding none of this rank's slots goes out as out 0, lse -inf,
        whatever the kernel left there (a select, so a NaN cannot survive)."""
        out = torch.where(valid[:, None, None], out, 0)
        lse = torch.where(valid[:, None], lse.float(), -math.inf)
        if self.over_sp:
            return _gather_partials(self.group, out, lse)
        heads = out.shape[1] // self.size
        blocks = _pack(out, lse).unflatten(1, (self.size, heads)).transpose(0, 1)
        return _unpack(self.group.to_partners(blocks, self.partners), out.dtype)


def _pack(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A partial's ``out`` ``[..., D]`` in its own dtype and its lse ``[...]`` in fp32 as one byte
    tensor ``[..., D * itemsize + 4]``, so one collective carries both at the out's width."""
    lse = lse.float().unsqueeze(-1).contiguous()
    return torch.cat([out.contiguous().view(torch.uint8), lse.view(torch.uint8)], dim=-1)


def _unpack(packed: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``_pack``'s ``(out, lse)`` back, as views of ``packed``."""
    return packed[..., :-4].view(dtype), packed[..., -4:].view(torch.float32).squeeze(-1)


def _gather_partials(group: CommGroup, out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Every rank of ``group``'s ``(out, lse)`` for the same queries, stacked on a new first dim."""
    return _unpack(group.all_gather(_pack(out, lse).unsqueeze(0), dim=0), out.dtype)



class ContextParallelWrapper:
    """A step's attention over a pool whose positions are placed over ranks
    (``PagedKVConfig.context_parallel``), from a backend's two wrappers: ``attend`` gets this
    rank's partials over its own slots, and ``ContextParallel.attend`` exchanges and merges them.

    A step whose rows each query one token attends the pages after its write (``context`` a
    decode-kind wrapper, ``new`` None). Any other step is two attentions: the stored context, not
    causal, planned on its context-only indptrs; and the step's own tokens, over the keys the layer
    passes to ``run``, on ``new``, which needs no exchange. ``context`` is planned with
    ``page_size`` the pool's slots per page on this rank. Each wrapper takes whatever its backend's
    ``run`` does (``attend_context``, ``attend_new``); ``new`` plans as ``plan(qo_indptr, causal)``.
    """

    def __init__(
        self,
        context,
        new,
        device: torch.device,
        captured_tokens: int = 0,
    ):
        self.context = context
        # None for a one-token-a-row step
        self.new = new
        self._device = device
        self._qo_indptr_buf = None
        # per query token, whether its row holds any of this rank's slots. A captured wrapper's
        # holds its bucket's every token (``captured_tokens``) and is never reallocated, so a
        # replay reads the address it was captured with
        self._captured = captured_tokens > 0
        self._valid = torch.zeros(captured_tokens, dtype=torch.bool, device=device) if self._captured else None
        self._stager = PinnedStager(torch.bool)
        # an eager step with no stored context on any row (from the logical plan, so the same on
        # every rank): it attends its own tokens alone, with no context call or exchange
        self.context_empty = False

    @torch.compiler.disable
    def plan(self, placement, page_size: int, kv_out, causal: bool = True, dtype: torch.dtype = torch.bfloat16):
        """``kv_out``: the step's logical ``KVPlanOutput`` for this label; ``page_size`` the logical one.
        The context attention reads its indptrs for a one-token-a-row step, its stored context
        alone otherwise."""
        context = kv_out.cpu_indptrs
        if self.new is not None:
            context = context.context_only(page_size)
        local, valid_rows = placement.local_indptrs(context, page_size)
        # no row has a stored page: logical, so the same on every rank
        self.context_empty = not self._captured and self.new is not None and not int(context.paged_kv_indptr[-1])
        if self.new is None:
            self.context.plan(dtype=dtype, **local.to_kwargs_dict())
        else:
            self.context.plan(causal=False, dtype=dtype, **local.to_kwargs_dict())
            self.new.plan(context.qo_indptr, causal=causal)
        self._qo_indptr_buf = getattr(self.context, "_qo_indptr_buf", None)
        valid = np.repeat(valid_rows, np.diff(context.qo_indptr.numpy()))
        n = len(valid)
        if self._captured:
            assert n <= self._valid.shape[0], f"{n} query tokens on a wrapper captured for {self._valid.shape[0]}"
        elif self._valid is None or self._valid.shape[0] < n:
            self._valid = torch.zeros(max(n, 256), dtype=torch.bool, device=self._device)
        # a captured bucket's whole buffer, so its padding tokens read False rather than a stale flag;
        # an eager step reads its own tokens' alone
        self._stager.copy_(self._valid, valid, pad_value=False if self._captured else None)

    def attend(
        self, placement, queries: tuple[torch.Tensor, ...], kv_cache_layer: torch.Tensor,
        new_keys: tuple[torch.Tensor | None, ...],
    ) -> torch.Tensor:
        """Attention over the whole placed pool (``ContextParallel.attend``), in the query's dtype:
        ``queries`` and ``new_keys`` are the query parts and the step's own keys as this backend's
        ``run`` takes them (``new_keys`` read only on a step that isn't one token a row)."""
        if self.new is not None and any(key is None for key in new_keys):
            raise ValueError(
                "attention over a context-parallel KV pool attends a step's own tokens from the keys "
                "run is passed (k and v, or MLA's latents as k); this step has more than one token "
                "a row and got none"
            )
        context = None if self.context_empty else lambda *parts: self.attend_context(*parts, kv_cache_layer)
        new = None if self.new is None else self.attend_new(*queries, *new_keys)
        return placement.attend(queries, context, new).to(queries[0].dtype)

    @property
    def reads_step_kv(self) -> bool:
        """Whether the step attends the keys ``run`` is passed (``AttentionManager.reads_step_kv``):
        any step but one token a row."""
        return self.new is not None

    def attend_context(self, *query_and_kv: torch.Tensor):
        """``(out, lse, valid)`` over this rank's slots, from ``context.run``'s query and KV arguments;
        ``valid`` as ``exchange_partials`` takes it."""
        out, lse = self.context.run(*query_and_kv, return_lse=True)
        return out, lse, self._valid[: query_and_kv[0].shape[0]]

    def attend_new(self, *query_and_kv: torch.Tensor):
        return self.new.run(*query_and_kv, return_lse=True)


def bound_placement(resources, pool: str, config: PagedKVConfig) -> "ContextParallel | None":
    """The placement of ``pool`` an attention resource over it binds (``resources[pool].placement``),
    refusing one that doesn't match the pool's ``context_parallel``: unplaced, a wrapper would plan
    the logical page size over a cache of ``page_size / P``."""
    placement = resources[pool].placement
    if (placement is not None) != config.context_parallel:
        raise ValueError(
            f"KV pool {pool!r} {'sets' if config.context_parallel else 'does not set'} context_parallel, "
            f"but the resource bound for it {'has no' if placement is None else 'has a'} placement"
        )
    return placement


def merge(outs: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    """Attention over the union of disjoint key sets from each set's partial: ``outs`` ``[K, ..., D]``
    and natural-log ``lses`` ``[K, ...]``; fp32. A partial with lse -inf and a finite out adds
    nothing; every row has a key in some partial (its own token, on the paged path)."""
    weights = torch.softmax(lses.float(), dim=0)
    return (weights.unsqueeze(-1) * outs).sum(dim=0)


@torch.compiler.disable
def gather_merge(group: CommGroup, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A query replicated on every rank of ``group`` attending keys sharded across them, from this
    rank's ``out`` ``[..., H, D]`` and natural-log ``lse`` ``[..., H]`` over its share: one gather
    of both, then ``merge``. Moves query-sized results rather than the keys."""
    if group.world_size == 1:
        return out
    return merge(*_gather_partials(group, out, lse)).to(out.dtype)


def positions_over_sp(specs: Iterable[NodeResourceSpec], node: str, sp_size: int) -> bool:
    """Whether ``node``'s KV pool places positions over its SP ranks. Such a node shards heads
    over TP alone and runs replicated across SP, so the model loads without its SP group.

    Raises when the node has a paged pool that is placed and one that isn't."""
    if sp_size == 1:
        return False
    pools = {
        spec.resource_key: spec.config.context_parallel for spec in specs
        if node in spec.nodes and isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig)
    }
    if any(pools.values()) and not all(pools.values()):
        raise ValueError(
            f"node {node!r} places positions over its SP ranks, so every paged KV pool it uses must: "
            f"set context_parallel on {sorted(key for key, placed in pools.items() if not placed)} too"
        )
    return any(pools.values())
