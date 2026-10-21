"""Data-parallel replica choice: a new request goes to the replica serving the
fewest live requests, so concurrent requests spread over the replicas instead of
landing on one at random."""
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from mstar.conductor.conductor import Conductor  # noqa: E402


def _wg(group_id, ranks, instance_ranks=()):
    return SimpleNamespace(_group_id=group_id, ranks=list(ranks), _instance_ranks=[list(r) for r in instance_ranks])


def _conductor(worker_graphs):
    conductor = object.__new__(Conductor)
    conductor.worker_graphs = worker_graphs
    conductor.requests = {}
    return conductor


def _admit(conductor, rid):
    assignment = conductor._assign_worker_graphs_to_workers()
    conductor.requests[rid] = SimpleNamespace(worker_graph_to_workers=assignment)
    return assignment


def test_requests_alternate_over_dp_replicas():
    # two node groups replicated on ranks 0 and 1, one wg each
    conductor = _conductor({0: _wg(0, [0, 1]), 1: _wg(0, [0, 1])})
    first = _admit(conductor, "a")
    second = _admit(conductor, "b")
    # a group's wgs stay on one replica, and the second request takes the idle one
    assert first[0] == first[1] and second[0] == second[1]
    assert first[0] != second[0]
    del conductor.requests["a"]
    assert _admit(conductor, "c")[0] == first[0]


def test_lockstep_instances_are_balanced_whole():
    # one sp2 node replicated as two instances: ranks (0, 1) and (2, 3)
    conductor = _conductor({0: _wg(0, [0, 1, 2, 3], instance_ranks=[(0, 1), (2, 3)])})
    picks = {tuple(_admit(conductor, rid)[0]) for rid in ("a", "b")}
    assert picks == {("worker_0", "worker_1"), ("worker_2", "worker_3")}


def test_single_replica_needs_no_load_scan():
    conductor = _conductor({0: _wg(0, [3])})
    conductor.requests = None  # never read when there is nothing to choose
    assert conductor._assign_worker_graphs_to_workers() == {0: ["worker_3"]}
