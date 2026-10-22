## qwen35 (GPU 0)  base=6e9b686b5 perf/overnight-shared dirty=0  cand=76e63021c perf/ovn-integ-cand-f dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.583, 1.423, 1.498, 1.394 | 1.425, 1.396, 1.419, 1.682 | +0.40% |
| c1_tok_s | 614.5, 683.5, 649.2, 698.4 | 682.2, 697.5, 685.7, 580.5 | +0.01% |
| c1_ttft_ms | 12.72, 11.63, 12.11, 10.85 | 11.94, 10.96, 11.51, 11.92 | -2.07% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.615, 2.27, 2.293, 2.197 | 2.17, 2.172, 2.16, 2.63 | -2.59% |
| c32_tok_s | 1.139e+04, 1.309e+04, 1.292e+04, 1.36e+04 | 1.371e+04, 1.376e+04, 1.378e+04, 1.13e+04 | +3.03% |
| c32_ttft_p50_ms | 52.02, 43.85, 49.71, 39.88 | 44.16, 37.93, 40.92, 53.77 | -4.69% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.663, 3.764, 4.027, 3.575 | 3.622, 3.358, 3.708, 3.823 | -3.46% |
| c64_tok_s | 1.593e+04, 1.557e+04, 1.436e+04, 1.648e+04 | 1.614e+04, 1.735e+04, 1.576e+04, 1.537e+04 | +3.67% |
| c64_ttft_p50_ms | 90.58, 92.19, 97.45, 79.25 | 87.99, 84.95, 89.31, 89.66 | -2.11% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=61.4 cand=60.9
    iter_total: 2.949 -> 2.865
    worker.postprocess.event_sync: 0.101 -> 0.099
    worker.gpu_thread.prepare_inputs: 0.023 -> 0.025
    worker.gpu_thread.exec: 1.683 -> 1.549
    await_gpu: 0.759 -> 0.623
    worker.postprocess_batch: 0.557 -> 0.561
    worker.schedule: 0.010 -> 0.010
    worker.plan_thread.pre_plan: 1.112 -> 1.028
    submit_spec: 1.072 -> 1.065

Correctness: {"base_vs_cand_match": 8, "n": 8, "base_r0_vs_r1_match": 8, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": null}
