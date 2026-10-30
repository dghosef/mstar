"""Context parallelism: a paged KV pool's token positions placed over a group of ranks
that would otherwise hold the same KV heads.

Page ids stay logical and are the same on every rank, so allocation, prefix keys,
offload and admission need nothing new. Within a page, logical slot ``s`` belongs to
partner ``s % P`` and sits at its physical slot ``s // P``, so each rank stores
``page_size // P`` slots a page. ``PagedKVConfig.place`` resolves the placement.
"""

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.spec import NodeResourceSpec


class ContextParallel(NamedTuple):
    """This rank's place in its KV pool's placement."""

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
