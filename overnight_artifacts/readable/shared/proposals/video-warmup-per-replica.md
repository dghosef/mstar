# video: warm every data-parallel replica (api_server warmup_requests)

Commit: 158af1568 on perf/ovn-video (mstar/api_server/entrypoint.py + test/modular/test_startup_warmup.py).
Depends on: 17d61232f (least-loaded replica choice, shared/proposals/video-dp-least-loaded.md);
without it the concurrent copies are placed at random and may still collide.

Mechanism: `_run_warmup_requests` ran each spec once, so a DP deployment warmed one random replica.
Each spec now runs as N concurrent copies, N = max over node_groups of len(ranks)/(tp_size*sp_size)
(`_data_parallel_replicas`). Single-replica configs: N = 1, behaviour unchanged.

Evidence: Cosmos3-Edge configs/cosmos3_edge_dp2.yaml (2x H200), 6 sequential requests, no client warmup:
before [10.5, 6.3, 6.2, 6.2, 6.3, 6.3] s, after [6.2, 6.2, 6.3, 6.3, 6.1, 6.3] s; startup 66 -> 57 s.
Tests: test_startup_warmup.py (7 pass, 1 new).
Affects: only configs with warmup_requests and replicated node groups.

## Verdict (integrator 03:10): MERGED as a5fa2ec21 + 387ce372c (test-stub fix added by integrator)

- Review: `_data_parallel_replicas` is computed from static config (ranks / (tp*sp), max over groups), and the copies run concurrently so least-loaded placement (ca5324b95) puts one on each replica. Single-replica configs get N=1, the same path as before. A multi-rank group whose parallelism isn't tp/sp would get N>1, which only means an extra concurrent warmup, so it is harmless. Startup-only code: no per-request or per-step cost.
- Tests: the CPU Core run had one NEW failure, test_serving_metrics::test_metrics_start_after_the_warmup_requests_and_before_the_bind. Its monkeypatched `_run_warmup_requests` stand-in took two arguments and now gets three. I fixed the stub in 387ce372c. After that, test_serving_metrics + test_startup_warmup fail only on the pre-existing test_refusals_count... (prometheus_client missing, which fails on base too).
- Served check, Qwen3.5-0.8B DP2 (`ovn_qwen3_5_0.8b_dp2.yaml`, warmup_requests = 2 images), base = shared 7fd1e3589. The log shows "Running 2 warmup request(s) before binding, 2 at a time (one per replica)", and startup is the same (111 vs 117 s). Throughput, 4 rounds on a quiet box: c=64 -3.7%, c=32 -1.5%, c=1 +2.4%, all noise, since the change can't touch steady state. An earlier 2-round run (-8/-10%) overlapped a -n 24 pytest run and is discarded.
