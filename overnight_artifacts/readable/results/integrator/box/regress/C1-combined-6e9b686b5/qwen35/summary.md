## qwen35 (GPU 0)  base=55b0215a3 perf/overnight-shared dirty=0  cand=6e9b686b5 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.465, 1.418, 1.513, 1.684 | 1.511, 1.36, 1.463, 1.485 | -4.31% |
| c1_tok_s | 663.1, 688.3, 644.2, 579.1 | 645.6, 716.5, 665.6, 654.5 | +4.17% |
| c1_ttft_ms | 12.45, 10.21, 11.52, 12.74 | 11.22, 10.3, 11.67, 12.23 | -3.21% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.063, 2.213, 2.235, 2.539 | 2.098, 2.126, 2.146, 2.293 | -4.28% |
| c32_tok_s | 1.429e+04, 1.343e+04, 1.319e+04, 1.175e+04 | 1.41e+04, 1.4e+04, 1.383e+04, 1.29e+04 | +4.13% |
| c32_ttft_p50_ms | 46.98, 44.34, 48.85, 47.53 | 45.39, 38.71, 39.27, 42.68 | -11.54% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.64, 3.83, 3.702, 3.703 | 3.868, 3.559, 3.848, 3.647 | +0.31% |
| c64_tok_s | 1.605e+04, 1.539e+04, 1.578e+04, 1.584e+04 | 1.519e+04, 1.644e+04, 1.528e+04, 1.609e+04 | -0.09% |
| c64_ttft_p50_ms | 90.36, 86.44, 91.77, 88.91 | 89.35, 83.76, 88.07, 85.21 | -3.10% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=61.1 cand=62.1
    iter_total: 2.811 -> 3.022
    worker.postprocess.event_sync: 0.117 -> 0.100
    worker.gpu_thread.prepare_inputs: 0.024 -> 0.024
    worker.gpu_thread.exec: 1.563 -> 1.635
    await_gpu: 0.648 -> 0.756
    worker.postprocess_batch: 0.543 -> 0.542
    worker.schedule: 0.010 -> 0.010
    worker.plan_thread.pre_plan: 1.031 -> 1.171
    submit_spec: 1.033 -> 1.133

Correctness: {"base_vs_cand_match": 8, "n": 8, "base_r0_vs_r1_match": 8, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": null}
