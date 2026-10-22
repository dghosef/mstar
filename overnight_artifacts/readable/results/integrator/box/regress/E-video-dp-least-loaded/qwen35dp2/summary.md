## qwen35dp2 (GPU 2,3)  base=6e9b686b5 perf/overnight-shared dirty=0  cand=b82a72bbc perf/ovn-integ-cand-e dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.605, 1.541, 1.752, 1.388 | 1.444, 1.393, 1.403, 1.407 | -10.15% |
| c1_tok_s | 605, 632.2, 557.1, 702.9 | 673.2, 697.8, 694, 691.6 | +10.38% |
| c1_ttft_ms | 12.93, 12.15, 12.96, 10.27 | 11.96, 11.77, 10.73, 10.99 | -5.95% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.584, 2.37, 2.574, 2.172 | 1.946, 1.956, 1.922, 1.864 | -20.73% |
| c32_tok_s | 1.125e+04, 1.201e+04, 1.132e+04, 1.331e+04 | 1.397e+04, 1.43e+04, 1.463e+04, 1.569e+04 | +22.35% |
| c32_ttft_p50_ms | 28.99, 43.47, 41.9, 45.77 | 52.35, 49.25, 46.52, 44.37 | +20.21% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.829, 3.653, 3.833, 3.73 | 3.669, 3.753, 3.463, 3.682 | -3.19% |
| c64_tok_s | 1.481e+04, 1.593e+04, 1.506e+04, 1.56e+04 | 1.581e+04, 1.582e+04, 1.671e+04, 1.582e+04 | +4.49% |
| c64_ttft_p50_ms | 129.4, 79.19, 84.48, 78.86 | 84.3, 75.41, 79.64, 78.6 | -14.52% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=26.6 cand=31.2
    iter_total: 2.911 -> 2.234
    worker.postprocess.event_sync: 0.096 -> 0.074
    worker.gpu_thread.prepare_inputs: 0.023 -> 0.020
    worker.gpu_thread.exec: 1.803 -> 1.174
    await_gpu: 0.674 -> 0.430
    worker.postprocess_batch: 0.437 -> 0.410
    worker.schedule: 0.010 -> 0.020
    worker.plan_thread.pre_plan: 1.029 -> 0.780
    submit_spec: 1.258 -> 0.933
- worker_1: bs base=28.9 cand=30.4
    iter_total: 2.523 -> 2.387
    worker.postprocess.event_sync: 0.055 -> 0.083
    worker.gpu_thread.prepare_inputs: 0.022 -> 0.020
    worker.gpu_thread.exec: 1.773 -> 1.252
    await_gpu: 0.628 -> 0.499
    worker.postprocess_batch: 0.372 -> 0.415
    worker.schedule: 0.010 -> 0.010
    worker.plan_thread.pre_plan: 0.880 -> 0.880
    submit_spec: 1.120 -> 0.998

Correctness: {"base_vs_cand_match": 7, "n": 8, "base_r0_vs_r1_match": 8, "cand_r0_vs_r1_match": 6, "cand_outputs_seen_in_base": 8, "first_diff": {"prompt": 7, "common_prefix_chars": 374, "base": "Romeo and Juliet meet in the garden.\n    *   Climax: The night of the wedding.\n ", "cand": "The marriage of the two families.\n    *   Climax: The night of the wedding.\n    "}}
