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
        # this rank's slots in each row's last page, and its KV length
        slots = page_size // self.size
        last = np.clip((indptrs.paged_kv_last_page_len.numpy() - self.index + self.size - 1) // self.size, 0, slots)
        lens = np.where(npages > 0, (npages - 1) * slots + last, 0)
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
        """Attention over the whole pool for this rank's query parts (each ``[N, H, *]``), fp32
        (``new``'s out as is when ``partial`` is None). ``partial`` gives ``(out, lse, valid)`` over
        this rank's slots; ``new`` is the step's own tokens' ``(out, lse)`` this rank computes (under
        SP placement its zigzag chunks, lse -inf on the rest). Every rank runs the same collectives:
        under SP placement one all-gather, under TP placement two all-to-alls, or none when
        ``partial`` is None (no row has stored context), which every rank must agree on."""
        if self.over_sp:
            return merge(*_gather_partials(self.group, *self.share(queries, partial, new)))
        if partial is None:
            return new[0]
        outs, lses = self.exchange_partials(*partial(*self.exchange_queries(queries)))
        if new is not None:
            outs = torch.cat([outs, new[0].to(outs.dtype).unsqueeze(0)])
            lses = torch.cat([lses, new[1].float().unsqueeze(0)])
        return merge(outs, lses)

    def share(
        self,
        queries: tuple[torch.Tensor, ...],
        partial: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        new: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Under SP placement, this rank's ``(out, lse)`` for every query as ``attend`` gathers it:
        its slots' partial merged with its share of the step's own tokens. Each own token's keys
        are attended by one rank, so merging every rank's share is the whole attention."""
        parts = [] if partial is None else [_masked(*partial(*queries))]
        if new is not None:
            parts.append(new)
        if len(parts) == 1:
            return parts[0][0], parts[0][1].float()
        out, lse = _merge_with_lse(
            torch.stack([part[0].float() for part in parts]), torch.stack([part[1].float() for part in parts]),
        )
        return out.to(parts[0][0].dtype), lse

    def exchange_queries(self, queries: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Under TP placement, the query parts this rank attends its slots for, from its own (each
        ``[N, H, *]``): its partners hold different query heads of one KV head, so it gets every
        partner's, ``[N, P * H, *]`` partner-major, the parts sent as one tensor."""
        q = queries[0] if len(queries) == 1 else torch.cat(queries, dim=-1)
        blocks = q.unsqueeze(0).expand(self.size, *q.shape)
        q = self.group.to_partners(blocks, self.partners).transpose(0, 1).flatten(1, 2)
        return q.split([part.shape[-1] for part in queries], dim=-1)

    def exchange_partials(
        self, out: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Under TP placement, every partner's partial for this rank's heads, ``[P, N, H, D]`` in
        ``out``'s dtype and ``[P, N, H]`` fp32, to ``merge``; from this rank's ``out`` ``[N, H', D]``
        and natural-log ``lse`` ``[N, H']`` over its slots, for the heads ``exchange_queries``
        returned, masked by ``valid`` (``_masked``)."""
        out, lse = _masked(out, lse, valid)
        heads = out.shape[1] // self.size
        blocks = _pack(out, lse).unflatten(1, (self.size, heads)).transpose(0, 1)
        return _unpack(self.group.to_partners(blocks, self.partners), out.dtype)


def _masked(out: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A partial over this rank's slots with each query row ``valid`` (``[N]``) marks as holding none
    of them set to out 0, lse -inf, whatever the kernel left there (a select, so a NaN cannot
    survive); lse fp32."""
    return torch.where(valid[:, None, None], out, 0), torch.where(valid[:, None], lse.float(), -math.inf)


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
    """A step's attention over a placed pool from a backend's two wrappers, exchanged and merged
    by ``ContextParallel.attend``. A one-token-a-row step attends the pages after its write
    (``new`` None); any other step attends its stored context (not causal) on ``context`` and its
    own tokens, from the keys the layer passes to ``run``, on ``new``: under TP placement all of
    them for this rank's heads, under SP placement this rank's zigzag chunks (``ZigzagWrapper``).
    ``page_size`` is the logical one."""

    def __init__(
        self,
        placement: ContextParallel,
        page_size: int,
        dtype: torch.dtype,
        context,
        new,
        device: torch.device,
        batch_size: int | None = None,
        max_total_tokens: int | None = None,
    ):
        self._placement, self._page_size, self._dtype = placement, page_size, dtype
        self.context = context
        # None for a one-token-a-row step
        self.new = new
        self._device = device
        self._qo_indptr_buf = None
        # per query token, whether its row holds any of this rank's slots. A captured wrapper's
        # holds its bucket's every token (a row's at least, and the bucket's) and is never
        # reallocated, so a replay reads the address it was captured with
        captured_tokens = max(batch_size or 0, max_total_tokens or 0)
        self._captured = captured_tokens > 0
        self._valid = torch.zeros(captured_tokens, dtype=torch.bool, device=device) if self._captured else None
        self._stager = PinnedStager(torch.bool)
        # an eager step with no stored context on any row (from the logical plan, so the same on
        # every rank): it attends its own tokens alone, with no context call
        self.context_empty = False

    @torch.compiler.disable
    def plan(self, kv_out, causal: bool = True):
        """``kv_out``: the step's logical ``KVPlanOutput`` for this label. The context attention reads
        its indptrs for a one-token-a-row step, its stored context alone otherwise."""
        context = kv_out.cpu_indptrs
        if self.new is not None:
            context = context.context_only(self._page_size)
        local, valid_rows = self._placement.local_indptrs(context, self._page_size)
        # no row has a stored page: logical, so the same on every rank
        self.context_empty = not self._captured and self.new is not None and not int(context.paged_kv_indptr[-1])
        # not causal whatever the step: every query sees all of it (a one-token row's own key
        # included), and FA MLA's causal tile check reads each rank's KV lengths
        self.context.plan(causal=False, dtype=self._dtype, **local.to_kwargs_dict())
        if self.new is not None:
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
        self, queries: tuple[torch.Tensor, ...], kv_cache_layer: torch.Tensor,
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

        def context(*parts: torch.Tensor):
            # over this rank's slots, with the rows holding any of them
            out, lse = self.context.run(*parts, kv_cache_layer, return_lse=True)
            return out, lse, self._valid[: parts[0].shape[0]]

        new = None if self.new is None else self.new.run(*queries, *new_keys, return_lse=True)
        return self._placement.attend(queries, None if self.context_empty else context, new).to(queries[0].dtype)

    @property
    def reads_step_kv(self) -> bool:
        """Whether the step attends the keys ``run`` is passed (``AttentionManager.reads_step_kv``):
        any step but one token a row."""
        return self.new is not None


class ZigzagRows(NamedTuple):
    """This rank's share of a step's own-token attention under SP placement. Each row's ``T``
    tokens are cut into ``2P`` chunks, chunk ``i`` of ``T // 2P`` tokens and one more when
    ``i < T % 2P``. Rank ``r`` attends chunks ``r`` and ``2P - 1 - r``, so every rank does the same
    causal work, each chunk's queries over its row's tokens up to the chunk's end. Arrays run over
    the non-empty chunks, row by row; token indices are the step's."""

    # [M] the queries this rank attends, chunk by chunk
    tokens: np.ndarray
    # [S + 1] each chunk's queries in ``tokens``
    qo_indptr: np.ndarray
    # [S + 1] each chunk's keys in ``kv_tokens``
    kv_indptr: np.ndarray
    # [K] each chunk's keys: its row's first token to the chunk's last
    kv_tokens: np.ndarray


def zigzag_rows(qo_indptr: np.ndarray, size: int, index: int) -> ZigzagRows:
    """Rank ``index`` of ``size``'s zigzag chunks of the rows ``qo_indptr`` lays out."""
    starts, lens = qo_indptr[:-1].astype(np.int64), np.diff(qo_indptr).astype(np.int64)
    chunks = 2 * size
    base, rem = lens // chunks, lens % chunks

    def bounds(chunk: int) -> tuple[np.ndarray, np.ndarray]:
        first = chunk * base + np.minimum(chunk, rem)
        return first, first + base + (chunk < rem)

    (lo, lo_end), (hi, hi_end) = bounds(index), bounds(chunks - 1 - index)
    first, end = np.stack([lo, hi], 1).ravel(), np.stack([lo_end, hi_end], 1).ravel()
    row = np.repeat(starts, 2)
    keep = end > first
    first, end, row = first[keep], end[keep], row[keep]
    q_lens = end - first
    return ZigzagRows(_ranges(row + first, q_lens), _indptr(q_lens), _indptr(end), _ranges(row, end))


def _indptr(lens: np.ndarray) -> np.ndarray:
    out = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=out[1:])
    return out


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Every ``arange(start, start + len)``, concatenated."""
    offsets = _indptr(lens)
    return np.repeat(starts - offsets[:-1], lens) + np.arange(offsets[-1])


class ZigzagWrapper:
    """A step's own tokens under SP placement, attended for this rank's zigzag chunks alone
    (``zigzag_rows``), as ``ContextParallel.share`` takes them: ``[N, H, D]`` and lse ``[N, H]``,
    0 and -inf off its chunks.

    The kernel is the backend's paged prefill wrapper over pages of one slot, page ``i`` the step's
    token ``i``, viewed from the keys ``run`` is passed. A chunk's keys are then a list of page
    indices, so a rank's two chunks, whose key prefixes overlap, plan as two rows of one call with
    no copy of the keys. ``run`` gathers the chunks' queries and scatters their outputs back.

    ``paged`` builds the kernel, given ``batch_size``, ``max_total_tokens`` and ``max_num_pages`` for
    a captured one; ``as_pages`` makes the one-slot pages from ``run``'s keys; the first
    ``num_queries`` of ``run``'s arguments are query parts, the rest keys."""

    def __init__(
        self,
        placement: ContextParallel,
        paged: Callable,
        as_pages: Callable[..., torch.Tensor],
        num_queries: int,
        device: torch.device,
        dtype: torch.dtype,
        captured_rows: int = 0,
        captured_tokens: int = 0,
    ):
        self._placement = placement
        self._as_pages = as_pages
        self._num_queries = num_queries
        self._device, self._dtype = device, dtype
        self._stager = PinnedStager(torch.int64)
        self._captured = captured_tokens > 0
        # how many queries the last plan gave this rank
        self._planned = 0
        self._tokens = None
        if not self._captured:
            self.kernel = paged()
            return
        # fixed for the bucket: at most 2 chunks a row, of at most ceil(T / 2P) tokens each, whose
        # keys are at most 2T; and one more row for the padding queries
        p = placement.size
        self._segments = 2 * captured_rows + 1
        self._queries = min(captured_tokens, -(-(captured_tokens + captured_rows * (2 * p - 1)) // p))
        self.kernel = paged(
            batch_size=self._segments, max_total_tokens=self._queries,
            max_num_pages=2 * captured_tokens + self._queries,
        )
        # a padding entry scatters to the row past the step's last token, which ``run`` drops
        self._pad = captured_tokens
        self._tokens = torch.full((self._queries,), captured_tokens, dtype=torch.int64, device=device)

    @torch.compiler.disable
    def plan(self, qo_indptr: torch.Tensor, causal: bool = True) -> None:
        rows = zigzag_rows(qo_indptr.numpy(), self._placement.size, self._placement.index)
        qo, kv, kv_tokens = rows.qo_indptr, rows.kv_indptr, rows.kv_tokens
        self._planned = len(rows.tokens)
        if self._captured:
            # a captured plan takes the batch size it was built for, and FlashInfer's run the
            # planned query count: the padding queries are one row, attending as many copies of
            # token 0 causally, and the rows left over hold no query and one key
            pad_queries, empty = self._queries - int(qo[-1]), self._segments - len(qo)
            qo = np.concatenate([qo, [self._queries], np.full(empty, self._queries)])
            kv = np.concatenate([kv, kv[-1] + max(pad_queries, 1) + np.arange(empty + 1)])
            kv_tokens = np.concatenate([kv_tokens, np.zeros(int(kv[-1]) - len(kv_tokens), dtype=np.int64)])
            self._stager.copy_(self._tokens, rows.tokens, pad_value=self._pad)
        elif self._planned:
            self._tokens = self._stager.to_device(rows.tokens, self._device)
        else:
            # none of the step's tokens is this rank's to attend
            return
        self.kernel.plan(
            qo_indptr=torch.from_numpy(qo.astype(np.int32)),
            paged_kv_indptr=torch.from_numpy(kv.astype(np.int32)),
            paged_kv_indices=torch.from_numpy(kv_tokens.astype(np.int32)),
            paged_kv_last_page_len=torch.ones(len(qo) - 1, dtype=torch.int32),
            kv_lens=torch.from_numpy(np.diff(kv).astype(np.int32)).as_subclass(HostLens),
            causal=causal, dtype=self._dtype,
        )

    def run(self, *queries_and_keys: torch.Tensor, return_lse: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        assert return_lse, "a rank's share of the own tokens is only ever merged"
        queries, keys = queries_and_keys[: self._num_queries], queries_and_keys[self._num_queries:]
        n, heads = queries[0].shape[:2]
        # one row past the step's, for padding entries to land in
        out = queries[0].new_zeros(n + 1, heads, queries[0].shape[-1])
        lse = torch.full((n + 1, heads), -math.inf, dtype=torch.float32, device=queries[0].device)
        if self._captured or self._planned:
            tokens = self._tokens
            own = [q.index_select(0, tokens.clamp(max=n - 1)) for q in queries]
            got, got_lse = self.kernel.run(*own, self._as_pages(*keys), return_lse=True)
            tokens = tokens.clamp(max=n)
            out.index_copy_(0, tokens, got.to(out.dtype))
            lse.index_copy_(0, tokens, got_lse.float())
        return out[:n], lse[:n]


def merge(outs: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    """Attention over the union of disjoint key sets from each set's partial: ``outs`` ``[K, ..., D]``
    and natural-log ``lses`` ``[K, ...]``; fp32, one ``[..., D]`` accumulated a part at a time. A
    partial with lse -inf and a finite out adds nothing, and a query no partial has a key for (a
    captured bucket's padding) comes out 0."""
    lses = lses.float()
    # finite on a row with no key, so its every weight is 0
    top = lses.amax(0).clamp_min(torch.finfo(torch.float32).min)
    weights = (lses - top).exp_()
    out = weights[0].unsqueeze(-1) * outs[0]
    for weight, part in zip(weights[1:], outs[1:], strict=True):
        out.addcmul_(weight.unsqueeze(-1), part)
    return out.div_(weights.sum(0).clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1))


def _merge_with_lse(outs: torch.Tensor, lses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``merge`` and the union's lse, fp32; a query no partial has a key for comes out 0, -inf."""
    lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - lse.masked_fill(lse == -math.inf, 0))
    return (weights.unsqueeze(-1) * outs).sum(dim=0), lse


@torch.compiler.disable
def gather_merge(group: CommGroup, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A query replicated on every rank of ``group`` attending keys sharded across them, from this
    rank's ``out`` ``[..., H, D]`` and natural-log ``lse`` ``[..., H]`` over its share: one gather
    of both, then ``merge``. Moves query-sized results rather than the keys."""
    if group.world_size == 1:
        return out
    return merge(*_gather_partials(group, out, lse)).to(out.dtype)


def _paged_pools(specs: Iterable[NodeResourceSpec], node: str) -> dict[str, bool]:
    """``node``'s paged KV pools, each to whether it places positions over the node's ranks.
    Raises when one is placed and another isn't."""
    pools = {
        spec.resource_key: spec.config.context_parallel for spec in specs
        if node in spec.nodes and isinstance(spec, KVSpec) and isinstance(spec.config, PagedKVConfig)
    }
    if any(pools.values()) and not all(pools.values()):
        raise ValueError(
            f"node {node!r} places positions over its SP ranks, so every paged KV pool it uses must: "
            f"set context_parallel on {sorted(key for key, placed in pools.items() if not placed)} too"
        )
    return pools


def positions_over_sp(specs: Iterable[NodeResourceSpec], node: str, sp_size: int) -> bool:
    """Whether ``node``'s KV pool places positions over its SP ranks. Such a node shards heads
    over TP alone and runs replicated across SP, so the model loads without its SP group.

    Raises when the node has a paged pool that is placed and one that isn't."""
    return sp_size > 1 and any(_paged_pools(specs, node).values())


def sp_rank_placement(specs: Iterable[NodeResourceSpec], node: str, sp_size: int, sp_enabled_nodes) -> bool:
    """Whether ``node``, on ``sp_size`` > 1 SP ranks, places its KV pools' positions over them
    (`positions_over_sp`) rather than running the model's own sequence parallelism (it is in
    ``sp_enabled_nodes``). Raises when it does neither, or both: the placement loads the node
    without the SP group its own sequence parallelism needs."""
    pools = _paged_pools(specs, node)
    placed = any(pools.values())
    if placed and node in sp_enabled_nodes:
        raise ValueError(
            f"node {node!r} runs its own sequence parallelism (sp_enabled_nodes), and a paged KV pool over "
            "it sets context_parallel, which would replace it: turn context_parallel off, or use sp_size 1 "
            "to place positions over TP ranks"
        )
    if not placed and node not in sp_enabled_nodes:
        raise ValueError(
            f"node {node!r} on {sp_size} SP ranks is not in sp_enabled_nodes {sorted(sp_enabled_nodes)}"
            + (": set context_parallel on its paged KV pools to place positions over the SP ranks instead"
               if pools else ", and holds no paged KV pool to place over them: put it in a node group of "
               "its own with sp_size 1")
        )
    return placed
