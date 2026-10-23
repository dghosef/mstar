## qwen35 (GPU 0)  base=d1c91505a perf/overnight-shared dirty=0  cand=bec51e1c5 perf/ovn-integ-cand-j dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.416, 1.378, 1.545, 1.426 | 1.614, 1.542, 1.666, 1.436 | +8.55% |
| c1_tok_s | 686.1, 705.8, 631.7, 683.1 | 603.6, 631.7, 585.5, 678.2 | -7.67% |
| c1_ttft_ms | 11.56, 11.28, 11.1, 11.11 | 12.45, 12.24, 12.3, 11.32 | +7.22% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.264, 2.158, 2.188, 2.109 | 2.373, 2.323, 2.577, 2.277 | +9.53% |
| c32_tok_s | 1.318e+04, 1.375e+04, 1.36e+04, 1.404e+04 | 1.25e+04, 1.284e+04, 1.162e+04, 1.31e+04 | -8.28% |
| c32_ttft_p50_ms | 42.67, 39.07, 41.67, 41.75 | 49.96, 42.76, 48.39, 41.59 | +10.62% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.64, 3.693, 3.795, 3.828 | 3.606, 3.491, 3.757, 3.589 | -3.43% |
| c64_tok_s | 1.598e+04, 1.591e+04, 1.551e+04, 1.539e+04 | 1.624e+04, 1.679e+04, 1.563e+04, 1.636e+04 | +3.58% |
| c64_ttft_p50_ms | 89.33, 85.51, 84.11, 85.14 | 87.45, 81.76, 88.43, 83.39 | -0.89% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=61.4
    iter_total: 1.692 -> 2.934
    worker.postprocess.event_sync: 0.037 -> 0.103
    worker.gpu_thread.prepare_inputs: 0.020 -> 0.026
    worker.gpu_thread.exec: 0.900 -> 1.667
    await_gpu: 0.350 -> 0.700
    worker.postprocess_batch: 0.223 -> 0.552
    worker.schedule: 0.040 -> 0.010
    worker.plan_thread.pre_plan: 0.630 -> 1.066
    submit_spec: 0.850 -> 1.044

Correctness: {"base_vs_cand_match": 8, "n": 8, "base_r0_vs_r1_match": 8, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": null}
