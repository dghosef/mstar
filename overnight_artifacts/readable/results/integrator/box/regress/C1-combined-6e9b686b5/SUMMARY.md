# Regression suite C1-combined-6e9b686b5  (2026-10-21 07:20:23)

## kokoro (GPU 3)  base=55b0215a3 perf/overnight-shared dirty=0  cand=6e9b686b5 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.005985, 0.005821 | 0.00628, 0.007061 | +13.01% |
| c1_ttfa_p50_ms | 29.48, 27.78 | 29.91, 33.46 | +10.68% |
| c32_audio_s_per_s | 681.8, 805.1 | 666.3, 652.5 | -11.31% |
| c32_audio_s_total | 756.4, 756.4 | 756.4, 756.4 | +0.00% |
| c32_failed | 0, 0 | 0, 0 |  |
| c32_rtf_p50 | 0.0428, 0.036 | 0.04376, 0.0448 | +12.38% |
| c32_ttfa_p50_ms | 202.7, 171 | 204.8, 218.7 | +13.33% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=5.2 cand=5.4
    iter_total: 38.990 -> 33.270
    worker.postprocess.event_sync: 12.300 -> 16.670
    worker.gpu_thread.prepare_inputs: 0.280 -> 0.290
    worker.gpu_thread.exec: 26.070 -> 20.960
    await_gpu: 20.570 -> 15.200
    worker.postprocess_batch: 15.040 -> 18.130
    worker.schedule: 0.020 -> 0.010

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.009411764705882352, "wer_cand_n": 32, "same_length_frac": 1.0, "waveform_snr_db_median": 19.35742950439453}

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

## qwen3tts (GPU 1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=6e9b686b5 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 100.4, 100.4 | 100.3, 102 | +0.72% |
| c16_audio_s_total | 433.5, 437.1 | 432.4, 432 | -0.72% |
| c16_failed | 0, 0 | 0, 0 |  |
| c16_rtf_p50 | 0.1575, 0.1547 | 0.157, 0.1561 | +0.29% |
| c16_ttfa_p50_ms | 75.72, 65.4 | 59.52, 64.42 | -12.18% |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.06225, 0.06076 | 0.06139, 0.06137 | -0.20% |
| c1_ttfa_p50_ms | 19.29, 18.54 | 18.39, 18.86 | -1.52% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=12.1 cand=11.3
    iter_total: 7.735 -> 7.308
    worker.postprocess.event_sync: 2.296 -> 2.530
    worker.gpu_thread.prepare_inputs: 1.380 -> 1.413
    worker.gpu_thread.exec: 2.205 -> 1.423
    await_gpu: 0.149 -> 0.128
    worker.postprocess_batch: 4.100 -> 3.915
    worker.schedule: 0.010 -> None
    worker.plan_thread.pre_plan: 0.699 -> 0.731
    submit_spec: 2.138 -> 2.448

Correctness: {"wer_base": 0.011764705882352941, "wer_base_n": 32, "wer_cand": 0.01411764705882353, "wer_cand_n": 32, "same_length_frac": 0.0625, "waveform_snr_db_median": -2.9027621746063232}

## zimage (GPU 2)  base=55b0215a3 perf/overnight-shared dirty=0  cand=6e9b686b5 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.078, 1.079 | 1.078, 1.092 | +0.64% |
| failed | 0, 0 | 0, 0 |  |
| lat_mean_s | 0.8587, 0.8549 | 0.8619, 0.8554 | +0.22% |
| lat_min_s | 0.8518, 0.8496 | 0.8545, 0.85 | +0.18% |
| lat_p50_s | 0.8561, 0.8546 | 0.8609, 0.8532 | +0.20% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.744 -> 71.006
    worker.postprocess.event_sync: 71.582 -> 71.810
    worker.gpu_thread.prepare_inputs: 0.231 -> 0.418
    worker.gpu_thread.exec: 3.238 -> 3.146
    await_gpu: 2.534 -> 2.414
    worker.postprocess_batch: 72.000 -> 72.122
    worker.schedule: 0.030 -> 0.020
    submit_spec: 0.494 -> 0.504

Correctness: {"psnr_base_vs_cand_r0_min": 31.739912812352035, "psnr_base_vs_cand_r0_median": 35.861414799167875, "psnr_base_r0_vs_r1_min": 31.08090778680339, "psnr_base_r0_vs_r1_median": 37.621791898753095, "psnr_cand_r0_vs_r1_min": 29.751755917346575, "psnr_cand_r0_vs_r1_median": 33.96804953990241}
