# world: warmup_requests `concurrency: N` (compile batched shapes before the bind)

- Branch `perf/ovn-world`, commits "[API server] warmup_requests: concurrency, ..." and
  "cosmos3_edge_droid.yaml: warm the batched sizes at the documented 32-step chunk" (on top of shared 387ce372c).
  Files: `mstar/api_server/entrypoint.py` (`WarmupRequest.concurrency`, validated positive int;
  `_run_warmup_requests` submits `copies * concurrency` at once), `test/modular/test_startup_warmup.py`
  (new test), `docs/serving.rst` (row text), `configs/cosmos3_edge_droid.yaml` (entries for 2..8).
- Why: the Cosmos3 cross-request action batch is now compiled (`[Cosmos3][Perf] Compile the cross-request
  action denoise batch`), and dynamo compiles one graph per batch size. Without a warmup, the first batch
  of each size stalls everyone (edge, 8 in flight after 4: p90 12.2-12.6 s, cold and warm inductor cache).
- Evidence so far (H200 ovn-world): cosmos_edge_warm round 1: warm arm 8-in-flight p90 1603 ms vs
  compile-only 1660 ms (the compile-only server had already compiled in its first round). With the
  chunk-32 warmups (cosmos_edge_warm2, first session on a fresh server, 8 in flight): 3.12 -> 5.33
  chunks/s, p90 3.7 -> 2.3 s. Startup 30 -> 42 s.
- Affects only configs that set `concurrency`; default 1 = the existing path. Startup-only code.
- Tests: `test_startup_warmup.py` not yet run on a box (no pytest on the Mac); please run it with the merge.

## Verdict (integrator 12:45): MERGED into perf/overnight-shared as 32db55b6c + a51da9261 (the cosmos3_edge_droid.yaml warmups)

- Review: `concurrency` is validated as a positive int (bools rejected) and defaults to 1, which is the existing path. It multiplies the per-replica `copies` from a5fa2ec21, so on DP deployments each replica sees `concurrency` concurrent copies (least-loaded placement). It is startup-only, with no request-path cost, and the serving.rst row is updated. One nit: `repeat` and `concurrency` both exist; a line in serving.rst on how they combine would help.
- Tests: test_startup_warmup.py (including the new test) passes on the H200 box, inside the 168-test GPU run above. Head vs main CPU suite: identical failure sets.
- The config change only touches cosmos3_edge_droid.yaml (world's). Its startup and latency effect is the proposer's measurement (8 in flight p90 3.7 -> 2.3 s, startup 30 -> 42 s).
