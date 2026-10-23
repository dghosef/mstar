# Regression suite FINAL-a51da9261  (2026-10-21 20:03:57)

## qwen35 (GPU 0)  base=55b0215a3 HEAD dirty=0  cand=a51da9261 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 2.182, 1.572, 1.376, 1.691 | 1.403, 1.422, 1.419, 1.473 | -16.18% |
| c1_tok_s | 446.8, 621.2, 706.8, 576.6 | 694.2, 684, 681.6, 661.2 | +15.72% |
| c1_ttft_ms | 16.71, 11.18, 11.23, 13.2 | 10.95, 11.33, 13.39, 11.28 | -10.27% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.83, 2.476, 2.158, 2.461 | 2.143, 2.201, 2.156, 2.234 | -11.99% |
| c32_tok_s | 1.058e+04, 1.211e+04, 1.378e+04, 1.206e+04 | 1.387e+04, 1.353e+04, 1.36e+04, 1.329e+04 | +11.87% |
| c32_ttft_p50_ms | 47.29, 42.11, 39.4, 48.16 | 41.46, 41.1, 49.82, 39.66 | -2.78% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.752, 3.824, 3.533, 4.159 | 3.361, 3.272, 3.485, 3.38 | -11.59% |
| c64_tok_s | 1.565e+04, 1.534e+04, 1.648e+04, 1.405e+04 | 1.742e+04, 1.775e+04, 1.679e+04, 1.712e+04 | +12.30% |
| c64_ttft_p50_ms | 90.79, 88.37, 90.04, 99.92 | 77.19, 84.92, 81.96, 92.75 | -8.75% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=60.7 cand=62.4
    iter_total: 2.780 -> 2.902
    worker.postprocess.event_sync: 0.102 -> 0.100
    worker.gpu_thread.prepare_inputs: 0.022 -> 0.026
    worker.gpu_thread.exec: 1.525 -> 1.781
    await_gpu: 0.660 -> 0.816
    worker.postprocess_batch: 0.570 -> 0.524
    worker.schedule: 0.020 -> 0.020
    worker.plan_thread.pre_plan: 1.018 -> 1.092
    submit_spec: 0.983 -> 1.063

Correctness: {"base_vs_cand_match": 8, "n": 8, "base_r0_vs_r1_match": 8, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": null}

## zimage (GPU 1)  base=55b0215a3 HEAD dirty=0  cand=a51da9261 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.128, 1.086, 1.087, 1.096 | 1.115, 1.103, 1.136, 1.098 | +1.25% |
| failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| lat_mean_s | 0.8583, 0.8504, 0.8531, 0.8572 | 0.8533, 0.8524, 0.849, 0.8537 | -0.31% |
| lat_min_s | 0.8473, 0.8445, 0.8438, 0.8471 | 0.8444, 0.8495, 0.8373, 0.8451 | -0.19% |
| lat_p50_s | 0.8571, 0.8491, 0.8504, 0.8591 | 0.8522, 0.8516, 0.8501, 0.8553 | -0.19% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.366 -> 70.318
    worker.postprocess.event_sync: 71.240 -> 70.884
    worker.gpu_thread.prepare_inputs: 0.231 -> 0.274
    worker.gpu_thread.exec: 3.312 -> 3.129
    await_gpu: 2.476 -> 2.422
    worker.postprocess_batch: 71.666 -> 71.412
    worker.schedule: 0.030 -> 0.030
    submit_spec: 0.484 -> 0.590

Correctness: {"psnr_base_vs_cand_r0_min": 32.30935983245371, "psnr_base_vs_cand_r0_median": 34.61973779938907, "psnr_base_r0_vs_r1_min": 32.24601555098156, "psnr_base_r0_vs_r1_median": 36.285502682794984, "psnr_cand_r0_vs_r1_min": 32.39115016474034, "psnr_cand_r0_vs_r1_median": 34.379795999238894}
