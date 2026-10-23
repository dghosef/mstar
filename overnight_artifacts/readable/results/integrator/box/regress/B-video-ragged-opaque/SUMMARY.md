# Regression suite B-video-ragged-opaque  (2026-10-21 06:57:11)

## qwen35 (GPU 2)  base=55b0215a3 perf/overnight-shared dirty=0  cand=b53845708 perf/ovn-integ-cand-b dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.648, 1.49, 1.44, 1.394 | 1.491, 1.411, 1.504, 1.481 | -1.43% |
| c1_tok_s | 590, 654.4, 676.4, 698.1 | 655, 690.1, 649, 658 | +1.27% |
| c1_ttft_ms | 13.64, 11.29, 11.24, 11.12 | 10.69, 11.23, 10.81, 11.15 | -7.21% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.623, 2.21, 2.217, 2.159 | 2.219, 2.141, 2.377, 2.16 | -3.38% |
| c32_tok_s | 1.13e+04, 1.345e+04, 1.352e+04, 1.388e+04 | 1.342e+04, 1.393e+04, 1.261e+04, 1.377e+04 | +3.02% |
| c32_ttft_p50_ms | 50.41, 42.27, 38.34, 36.23 | 43.72, 38.43, 35.65, 42.11 | -4.39% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.863, 3.722, 3.667, 3.658 | 3.796, 3.883, 3.736, 3.457 | -0.25% |
| c64_tok_s | 1.497e+04, 1.58e+04, 1.602e+04, 1.606e+04 | 1.546e+04, 1.51e+04, 1.582e+04, 1.699e+04 | +0.83% |
| c64_ttft_p50_ms | 108.8, 82.85, 84.14, 83.84 | 88.68, 92.02, 80.3, 80.49 | -5.04% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=61.7 cand=61.9
    iter_total: 3.146 -> 3.007
    worker.postprocess.event_sync: 0.111 -> 0.107
    worker.gpu_thread.prepare_inputs: 0.026 -> 0.028
    worker.gpu_thread.exec: 1.719 -> 1.710
    await_gpu: 0.741 -> 0.780
    worker.postprocess_batch: 0.594 -> 0.562
    worker.schedule: 0.015 -> 0.010
    worker.plan_thread.pre_plan: 1.184 -> 1.148
    submit_spec: 1.185 -> 1.127

Correctness: {"base_vs_cand_match": 6, "n": 8, "base_r0_vs_r1_match": 6, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": {"prompt": 6, "common_prefix_chars": 190, "base": "in Python Code Quality:**\n    *   Readability (readability, clarity, maintainabi", "cand": "of Clean Python Code:**\n    *   Readability (readability, clarity, maintainabili"}}

## zimage (GPU 3)  base=55b0215a3 perf/overnight-shared dirty=0  cand=b53845708 perf/ovn-integ-cand-b dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.13, 1.092 | 1.074, 1.09 | -2.59% |
| failed | 0, 0 | 0, 0 |  |
| lat_mean_s | 0.8601, 0.8588 | 0.8687, 0.855 | +0.28% |
| lat_min_s | 0.8496, 0.8505 | 0.8579, 0.8484 | +0.36% |
| lat_p50_s | 0.8585, 0.8567 | 0.864, 0.8544 | +0.19% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.558 -> 70.594
    worker.postprocess.event_sync: 71.350 -> 71.264
    worker.gpu_thread.prepare_inputs: 0.308 -> 0.261
    worker.gpu_thread.exec: 3.158 -> 3.302
    await_gpu: 2.406 -> 2.554
    worker.postprocess_batch: 71.744 -> 71.834
    worker.schedule: 0.030 -> 0.030
    submit_spec: 0.545 -> 0.572

Correctness: {"psnr_base_vs_cand_r0_min": 30.34426668121752, "psnr_base_vs_cand_r0_median": 35.50585759032445, "psnr_base_r0_vs_r1_min": 28.309505069203937, "psnr_base_r0_vs_r1_median": 33.333071615898554, "psnr_cand_r0_vs_r1_min": 31.51272900530323, "psnr_cand_r0_vs_r1_median": 36.207936950022344}
