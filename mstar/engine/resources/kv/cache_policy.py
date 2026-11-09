"""A paged KV pool's prefix-cache policy: which cached page it evicts first, which waiting request passes the first
one in the queue first, and which victim moves off the device first. ``cache_policy`` in the pool's ``resources:``
block: ``lru`` (the default), ``lfu``, ``cache_aware``, or ``"package.module:Class"`` naming a subclass of
`CachePolicy`.

What a policy may choose, and what it may not:

- **Eviction** orders only the pages the index alone holds. Every rank evicts, so the key must be the same on every
  rank: computed from its arguments alone, with no clock, randomness or state of its own. The conductor's routing
  mirror (`KVRoutingMirror`) evicts with the same policy.
- **Passing** orders only the requests the runner already lets pass the first request of a queue (A2). The pages a
  passer frees are kept for that first request whatever the order, so its wait stays bounded.
- **Victims** are only ever requests ordered after the request that runs short (`admission_policy.order`), so every
  request still finishes (docs/design/admission_and_preemption.md §5.5). The key orders them; `KVManager._choose`
  still decides which go to the host, as the host's room allows in that order, and leaves out any whose move adds
  nothing.

Pass and victim keys are read on the rank that decides (rank 0 at TP), which journals what it decided.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from mstar.utils.policy import resolve_policy


@dataclass(eq=False)
class VictimCost:
    """A request a move could take for another's shortage. ``place`` is free to read: 0 for the request ordered last,
    1 for the one before it, and so on. The others are worked out when first read, under the pool's lock."""
    rid: str
    place: int
    _frees: Callable[[], int]
    _fits_host: Callable[[], bool]
    _rebuild_tokens: Callable[[], int]

    @cached_property
    def frees(self) -> int:
        """The room its move alone would make: its own pages, the cached pages it alone kept from eviction, and what
        its allotment has unused."""
        return self._frees()

    @cached_property
    def fits_host(self) -> bool:
        """Whether the host's room holds all of it now, before any other victim takes from that room."""
        return self._fits_host()

    @cached_property
    def rebuild_tokens(self) -> int:
        """The tokens it holds on the device, which a drop would have to write again, less those of its prompt pages
        the index keeps."""
        return self._rebuild_tokens()


class CachePolicy:
    """``lru``: the least recently hit cached page goes first, and passers and victims go in the order the requests
    are in, the last request the first victim. This is the default."""

    def evict_key(self, clock: int, previous: Any) -> Any:
        """The eviction key of a cached page that is filed (``previous`` None) or hit (``previous`` its key until now)
        at index time ``clock``, which grows with every filing and every hit. The page with the lowest key goes first.
        The key must not fall when a page is hit, since the index re-sorts a page only when its key rises."""
        return clock

    def pass_key(self, hit_pages: int, need_pages: int) -> Any:
        """The key of a waiting request that may pass the first one in a queue, which holds ``hit_pages`` of the
        pages it asks for in cached pages and would take ``need_pages`` more. The lowest key tries first; equal keys
        go in the requests' order."""
        return 0

    def victim_key(self, victim: VictimCost) -> Any:
        """The key of a candidate victim. The lowest moves first; equal keys go last request first."""
        return victim.place


class LFU(CachePolicy):
    """``lfu``: the least often hit cached page goes first, the least recently hit among equals."""

    def evict_key(self, clock: int, previous: Any) -> Any:
        return (1 if previous is None else previous[0] + 1, clock)


class CacheAware(CachePolicy):
    """``cache_aware``: evicts as ``lru``. The passer with the most cached pages tries first, as SGLang's
    longest-prefix-match orders its queue. The victim that is cheapest to bring back moves first: one the host holds
    whole, then the one with the fewest tokens to write again, then the last request. A long request then tends to be
    dropped last, and a short one may be dropped each time a request runs short."""

    def pass_key(self, hit_pages: int, need_pages: int) -> Any:
        return -hit_pages

    def victim_key(self, victim: VictimCost) -> Any:
        return (not victim.fits_host, victim.rebuild_tokens, victim.place)


CACHE_POLICIES: dict[str, type[CachePolicy]] = {"lru": CachePolicy, "lfu": LFU, "cache_aware": CacheAware}


def cache_policy(spec: str) -> CachePolicy:
    """The policy ``spec`` names, as ``PagedKVConfig.cache_policy`` holds it."""
    return resolve_policy("cache_policy", spec, CACHE_POLICIES, CachePolicy)()
