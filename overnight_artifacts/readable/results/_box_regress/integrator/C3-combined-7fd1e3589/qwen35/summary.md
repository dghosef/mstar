## qwen35 (GPU 0)  base=55b0215a3 perf/overnight-shared dirty=0  cand=7fd1e3589 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.409, 1.528, 1.393, 1.44 | 1.436, 1.375, 1.421, 1.531 | -0.11% |
| c1_tok_s | 687.5, 638.3, 697.7, 676.4 | 678.5, 707.6, 686.1, 636 | +0.31% |
| c1_ttft_ms | 12.44, 11.28, 11.54, 11.31 | 11.01, 11.12, 10.78, 11.7 | -4.22% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.221, 2.229, 2.039, 2.192 | 2.165, 2.169, 2.201, 2.391 | +2.84% |
| c32_tok_s | 1.33e+04, 1.335e+04, 1.454e+04, 1.355e+04 | 1.373e+04, 1.373e+04, 1.351e+04, 1.252e+04 | -2.28% |
| c32_ttft_p50_ms | 48.96, 43.27, 43.33, 43.38 | 40.08, 39.48, 41.96, 38.81 | -10.40% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.871, 3.67, 3.751, 3.638 | 3.447, 3.486, 3.75, 3.506 | -4.95% |
| c64_tok_s | 1.509e+04, 1.599e+04, 1.575e+04, 1.606e+04 | 1.684e+04, 1.68e+04, 1.576e+04, 1.652e+04 | +4.81% |
| c64_ttft_p50_ms | 92.84, 87.65, 82.54, 89.43 | 90.32, 84.36, 79.47, 90.22 | -2.30% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=60.5 cand=61.8
    iter_total: 2.868 -> 2.752
    worker.postprocess.event_sync: 0.096 -> 0.098
    worker.gpu_thread.prepare_inputs: 0.024 -> 0.024
    worker.gpu_thread.exec: 1.566 -> 1.533
    await_gpu: 0.672 -> 0.684
    worker.postprocess_batch: 0.558 -> 0.522
    worker.schedule: 0.010 -> 0.010
    worker.plan_thread.pre_plan: 1.073 -> 1.026
    submit_spec: 1.043 -> 0.988

Correctness: {"base_vs_cand_match": 8, "n": 8, "base_r0_vs_r1_match": 8, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": null}
