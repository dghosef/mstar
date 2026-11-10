"""Where a prefill row keeps a snapshot of a request's recurrent state.

The kernel checkpoints the state every page of a row; the pool decides which of those boundaries it reserves a
snapshot slot for (`RecurrentStatePool._admit_prefix`). That choice is a policy, chosen by the serving config's
``resources: <pool>: checkpoints:`` (a built-in name, or ``"pkg.module:Class"``), and it must be deterministic over
the row it is given.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import NamedTuple


class CheckpointRow(NamedTuple):
    """What the pool knows of a prefill row when it reserves its snapshots. Boundaries are token counts of the
    request's stream, so a page multiple lies on the row's checkpoint grid when ``start`` does."""
    # the tokens the stream holds already (a restored snapshot's, or the chunks before this one)
    start: int
    # the tokens this row writes
    span: int
    # the tokens the KV cache hit, which may run past ``start``
    hit: int
    # the checkpoint interval of the step, a page multiple
    every: int
    # whether the prompt goes on past this row (a chunk short of its end)
    mid_prefill: bool
    # the token counts, page-floored, where the request's prompt breaks into messages (`prefix_marks`)
    marks: tuple[int, ...] = ()

    @property
    def end(self) -> int:
        """The row's last boundary: its end, or for a prompt's last row the grid boundary before its last token,
        which a resend of the prompt runs again. The pool treats a snapshot kept here as the row's end (the one
        that supersedes the earlier chunk's), whatever the policy."""
        return self.start + (self.span if self.mid_prefill else self.span - 1) // self.every * self.every


class CheckpointPolicy:
    """Which boundaries of a row keep a snapshot. The default keeps today's: the row's last boundary, and the
    KV hit's last boundary when it runs past the restored length."""

    # whether the policy reads `CheckpointRow.marks`, so the API server computes them for a request
    wants_marks = False

    def boundaries(self, row: CheckpointRow) -> Iterable[int]:
        return {row.end, row.start + (row.hit - row.start) // row.every * row.every}


class MessageBoundaries(CheckpointPolicy):
    """The default's, and the page floor of each message boundary of the prompt that the row crosses: a system
    block, or a long history, that other requests of a deployment share is then restored from the state at its
    end, and not recomputed by the first request that diverges inside it. A boundary off the row's checkpoint grid
    (a step that checkpoints every few pages) is left out."""

    wants_marks = True

    def boundaries(self, row: CheckpointRow) -> Iterable[int]:
        found = set(super().boundaries(row))
        found.update(m for m in row.marks if row.start < m < row.start + row.span and not (m - row.start) % row.every)
        return found


BUILT_IN = {"rows": CheckpointPolicy, "messages": MessageBoundaries}


def checkpoint_policy(name: str) -> CheckpointPolicy:
    """The built-in policy ``name``, or the class ``"pkg.module:Class"`` names."""
    if name in BUILT_IN:
        return BUILT_IN[name]()
    module, _, attr = name.partition(":")
    try:
        policy = getattr(importlib.import_module(module), attr)()
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"checkpoints {name!r} is neither a built-in ({sorted(BUILT_IN)}) nor a 'pkg.module:Class'"
        ) from exc
    if not isinstance(policy, CheckpointPolicy):
        raise ValueError(f"checkpoints {name!r} is not a CheckpointPolicy")
    return policy
