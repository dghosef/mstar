## qwen35 (GPU 1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=55b0215a3 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_itl_ms | 1.579, 1.461 | 1.871, 1.655 | +16.02% |
| c1_tok_s | 616.9, 666.4 | 521.7, 589.1 | -13.44% |
| c1_ttft_ms | 12.45, 11.46 | 13.46, 12.28 | +7.67% |
| c32_failed | 0, 0 | 0, 0 |  |
| c32_itl_ms | 2.645, 2.185 | 2.566, 2.694 | +8.92% |
| c32_tok_s | 1.13e+04, 1.359e+04 | 1.152e+04, 1.107e+04 | -9.25% |
| c32_ttft_p50_ms | 47.83, 43.89 | 57.85, 51.77 | +19.51% |
| c64_failed | 0, 0 | 0, 0 |  |
| c64_itl_ms | 4.337, 3.934 | 4.264, 3.886 | -1.47% |
| c64_tok_s | 1.358e+04, 1.5e+04 | 1.369e+04, 1.49e+04 | +0.01% |
| c64_ttft_p50_ms | 98.07, 86.72 | 108.6, 98.39 | +11.99% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=60.8 cand=61.9
    iter_total: 3.046 -> 3.134
    worker.postprocess.event_sync: 0.103 -> 0.099
    worker.gpu_thread.prepare_inputs: 0.027 -> 0.024
    worker.gpu_thread.exec: 1.592 -> 1.749
    await_gpu: 0.674 -> 0.723
    worker.postprocess_batch: 0.578 -> 0.590
    worker.schedule: 0.013 -> 0.020
    worker.plan_thread.pre_plan: 1.105 -> 1.107
    submit_spec: 1.170 -> 1.109

Correctness: {"base_vs_cand_match": 8, "n": 8, "base_r0_vs_r1_match": 8, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": null}
