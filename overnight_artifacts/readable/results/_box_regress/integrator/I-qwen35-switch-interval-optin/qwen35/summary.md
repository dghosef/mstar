## qwen35 (GPU 0)  base=d1c91505a perf/overnight-shared dirty=0  cand=3cd98153e perf/ovn-integ-cand-i dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.58, 1.547, 1.402, 1.484 | 1.482, 1.504, 1.374, 1.384 | -4.48% |
| c1_tok_s | 618.7, 632.4, 695.9, 655.5 | 658.8, 645.4, 710, 704 | +4.45% |
| c1_ttft_ms | 10.86, 9.983, 10.32, 12.1 | 10.7, 10.61, 10.22, 10.83 | -2.09% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.223, 2.382, 2.152, 2.606 | 2.15, 2.259, 2.094, 2.549 | -3.32% |
| c32_tok_s | 1.342e+04, 1.258e+04, 1.383e+04, 1.143e+04 | 1.382e+04, 1.327e+04, 1.42e+04, 1.175e+04 | +3.47% |
| c32_ttft_p50_ms | 40.9, 43.05, 42.52, 51.9 | 39.14, 36.31, 41.8, 42.26 | -10.58% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.746, 3.749, 3.831, 3.854 | 3.37, 3.418, 3.26, 3.526 | -10.58% |
| c64_tok_s | 1.575e+04, 1.579e+04, 1.544e+04, 1.523e+04 | 1.74e+04, 1.703e+04, 1.795e+04, 1.66e+04 | +10.90% |
| c64_ttft_p50_ms | 83.6, 79.63, 80.26, 89.39 | 79.58, 85.75, 79.38, 84.82 | -1.01% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=62.3 cand=61.3
    iter_total: 3.082 -> 2.926
    worker.postprocess.event_sync: 0.100 -> 0.100
    worker.gpu_thread.prepare_inputs: 0.022 -> 0.023
    worker.gpu_thread.exec: 1.606 -> 1.479
    await_gpu: 0.742 -> 0.721
    worker.postprocess_batch: 0.554 -> 0.533
    worker.schedule: 0.010 -> 0.015
    worker.plan_thread.pre_plan: 1.182 -> 1.161
    submit_spec: 1.162 -> 1.091

Correctness: {"base_vs_cand_match": 8, "n": 8, "base_r0_vs_r1_match": 8, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": null}
