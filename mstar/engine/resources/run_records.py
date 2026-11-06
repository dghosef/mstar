"""Run records: what a run's member step needs for the closer's rounds, by request.

A run of several nodes closes on one node (``step_repeats``); every other node
of its Loop, a member, runs the run's first iteration as an ordinary step. The
closer's step then replays each member's capture for the rest of the run, on a
slot of the member's it leases and stages again from this record, so the run
never depends on the member's own slot surviving other requests' steps.

The record is the member step as planned (its declared step, its capture key,
its batch size) and its preprocess output without the inputs the run carries
in, which the first round overwrites. A member's next step on the same rows and
walk replaces it, the closer's gather takes it, and the request's removal drops
it. Host state only: nothing on the device is held.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.spec import NodeResourceSpec


@dataclass
class RunRecordsSpec(NodeResourceSpec):
    """Declared by the engine for a node that is a run's member: a model writes none."""

    @property
    def resource_class(self) -> type[RunRecords]:
        return RunRecords


def records_key(node: str) -> str:
    return f"run_records:{node}"


@dataclass(frozen=True)
class RunRecord:
    """One member step of a run's first iteration."""
    # each row's request, by wire id: a handle is another request's once reused
    requests: tuple[str, ...]
    step: Any
    cg_key_info: Any
    graph_walk: str
    bs: int
    num_tokens: int | None
    # preprocess output, without the carried inputs
    staged: Mapping[str, Any]


class RunRecords(Resource):
    """A node's run records, by (rows, walk)."""

    def __init__(self) -> None:
        self._records: dict[tuple[tuple[Hashable, ...], str], RunRecord] = {}

    @classmethod
    def build(cls, spec: RunRecordsSpec, info: EngineResourceInfo) -> RunRecords:
        return cls()

    def keep(self, rows: Sequence[Hashable], record: RunRecord) -> None:
        self._records[(tuple(rows), record.graph_walk)] = record

    def take(self, rows: Sequence[Hashable], graph_walk: str, requests: Sequence[str]) -> RunRecord | None:
        """The record of these rows on this walk, if they are still its requests."""
        record = self._records.pop((tuple(rows), graph_walk), None)
        return record if record is not None and record.requests == tuple(requests) else None

    def remove_request(self, rid: Hashable) -> None:
        for key in [key for key in self._records if rid in key[0]]:
            del self._records[key]
