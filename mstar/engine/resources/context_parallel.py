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
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import CommGroup, _all_to_all_single
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens, PagedIndptrs
from mstar.engine.resources.spec import NodeResourceSpec


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
            out[new_kvp[row[keep]] + within[keep]] = indices.numpy()[keep]
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

    def exchange_queries(self, q: torch.Tensor) -> torch.Tensor:
        """The queries this rank attends its slots for, from its own ``q`` ``[N, H, D]``. SP partners
        hold the same heads, so that is ``q``; TP partners hold different query heads of one KV
        head, so each gets every partner's: ``[N, P * H, D]``, partner-major."""
        if self.over_sp:
            return q
        return self._to_partners(q.unsqueeze(0).expand(self.size, *q.shape)).transpose(0, 1).flatten(1, 2)

    def exchange_partials(
        self, out: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every partner's partial for this rank's heads, ``[P, N, H, D]`` and ``[P, N, H]`` fp32, to
        ``merge``; from this rank's ``out`` ``[N, H', D]`` and natural-log ``lse`` ``[N, H']`` over
        its slots, for the heads ``exchange_queries`` returned. A query row ``valid`` (``[N]``)
        marks as holding none of this rank's slots goes out as out 0, lse -inf, whatever the kernel
        left there (a select, so a NaN cannot survive)."""
        out = torch.where(valid[:, None, None], out.float(), 0.0)
        lse = torch.where(valid[:, None], lse.float(), -math.inf)
        packed = torch.cat([out, lse.unsqueeze(-1)], dim=-1)
        if self.over_sp:
            gathered = self.group.all_gather(packed.unsqueeze(0), dim=0)
        else:
            heads = packed.shape[1] // self.size
            gathered = self._to_partners(packed.unflatten(1, (self.size, heads)).transpose(0, 1))
        return gathered[..., :-1], gathered[..., -1]

    def _to_partners(self, blocks: torch.Tensor) -> torch.Tensor:
        """``blocks[j]`` to partner ``j``, and back the block each partner sent this rank, in partner
        order: one all-to-all over the whole group with nothing to the ranks outside the block."""
        blocks = blocks.contiguous()
        splits = [0] * self.group.world_size
        for partner in self.partners:
            splits[partner] = blocks[0].numel()
        recv = torch.empty_like(blocks)
        _all_to_all_single(self.group.device_group, blocks.view(-1), recv.view(-1), splits, splits)
        return recv


def merge(outs: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    """Attention over the union of disjoint key sets from each set's partial: ``outs`` ``[K, ..., D]``
    and natural-log ``lses`` ``[K, ...]``; fp32. A partial with lse -inf and a finite out adds
    nothing, and a row with no keys in any partial comes out 0."""
    lses = lses.float()
    top = lses.amax(dim=0)
    top = torch.where(top == -math.inf, 0.0, top)
    weights = torch.exp(lses - top)
    total = weights.sum(dim=0)
    out = (weights.unsqueeze(-1) * outs.float()).sum(dim=0)
    return out / torch.where(total > 0, total, 1.0).unsqueeze(-1)


@torch.compiler.disable
def gather_merge(group: CommGroup, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A query replicated on every rank of ``group`` attending keys sharded across them, from this
    rank's ``out`` ``[..., H, D]`` and natural-log ``lse`` ``[..., H]`` over its share: one gather
    of both, then ``merge``. Moves query-sized results rather than the keys."""
    if group.world_size == 1:
        return out
    packed = torch.cat([out.float(), lse.float().unsqueeze(-1)], dim=-1)
    gathered = group.all_gather(packed.unsqueeze(0), dim=0)
    return merge(gathered[..., :-1], gathered[..., -1]).to(out.dtype)


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
