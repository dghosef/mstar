# video: least-loaded data-parallel replica choice in the conductor

Commit: 17d61232f on perf/ovn-video (mstar/conductor/conductor.py + test/modular/test_conductor_replica_assignment.py).

Mechanism: `_assign_worker_graphs_to_workers` picked a node group's DP replica (or lockstep instance)
with `np.random.randint` per request. Now it picks the replica whose workers serve the fewest live
requests (`self.requests`), ties at random. The scan over live requests only runs when some group
has >1 replica, so single-replica configs pay nothing. Lockstep instances are still chosen whole.
Python only (the Rust side has no replica choice), so invariant 5 is unaffected.

Evidence (Wan2.2-TI2V-5B, configs/wan22_dp2.yaml = all nodes replicated on ranks [0,1], 2x H200,
1280x704x81, 20 steps, 8 requests closed loop; control tree = same code with base conductor.py):
- c=4: random 3.58 videos/min -> least-loaded 5.21 (+45%)
- c=2: random 5.02 and 3.31 (two runs; luck) -> least-loaded 4.98, 4.98
Ceiling for 2 replicas = 2 x 2.50 (1-GPU) = 5.0 at c=2.

Affects: any config with a node group on several ranks without tp/sp (DP replicas), e.g. anything
served with replicated node groups. Text LLMs with DP replicas should see better balance at
moderate concurrency; per-request cost is one pass over live requests when a choice exists.
Tests: test_conductor_replica_assignment.py (3 tests) passes.

## Verdict (integrator 01:05): MERGED into perf/overnight-shared as ca5324b95

- Review: the choice is made centrally in the conductor, once per request, and lockstep instances are still chosen whole, so ranks never disagree (invariant 11). The Rust graph runtime only consumes the conductor's assignment and has no replica choice of its own (checked rust/src/graph), so invariant 5 doesn't apply. CPU cost: one pass over live requests per new request, and only when a group has more than one replica. Single-replica deployments now return early and skip the old `np.random.randint`, a tiny saving. Nit: the scan is O(live requests x workers per request) per arrival. That is fine at hundreds of requests; if DP serving ever runs thousands of concurrent requests, keep a per-worker live count up to date instead.
- CPU tests (E+F together vs base 55b0215a3): no new failures (the known test_eval_compare flake only). The 3 new tests are in the run.
- Cross-model DP check I added: Qwen3.5-0.8B with the whole node group replicated on 2 H200s (`ovn_qwen3_5_0.8b_dp2.yaml`, harness dir), base = shared 6e9b686b5, 4 rounds ABBA, fresh servers:
  - c=32: 11.3-13.3k -> 14.0-15.7k tok/s (**+22%**). Per-replica batch at steady state 26.6/28.9 -> 31.2/30.4 (balanced), iter_total 2.91/2.52 -> 2.23/2.39 ms.
  - c=64: 14.8-15.9k -> 15.8-16.7k (+4.5%), TTFT p50 -14%. c=1: noise (there is nothing to balance).
  - Greedy: every cand output appears in base (8/8).
- No single-replica model is affected: none of the suite cells' default configs has a DP group, and those configs take the early return.
