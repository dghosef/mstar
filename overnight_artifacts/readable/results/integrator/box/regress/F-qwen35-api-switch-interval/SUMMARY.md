# Regression suite F-qwen35-api-switch-interval  (2026-10-21 08:07:22)

## kokoro (GPU 1)  base=6e9b686b5 perf/overnight-shared dirty=0  cand=76e63021c perf/ovn-integ-cand-f dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_rtf_p50 | 0.006405, 0.005868, 0.006089, 0.006191 | 0.005815, 0.006885, 0.006041, 0.006006 | +0.79% |
| c1_ttfa_p50_ms | 30.74, 28.41, 29.06, 30.02 | 27.77, 34.63, 28.82, 28.87 | +1.57% |
| c32_audio_s_per_s | 748.4, 781.1, 755.8, 799 | 739.4, 725.2, 769.4, 737 | -3.67% |
| c32_audio_s_total | 756.4, 756.4, 756.4, 756.4 | 756.4, 756.4, 756.4, 756.4 | +0.00% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_rtf_p50 | 0.03768, 0.03633, 0.0369, 0.03682 | 0.03882, 0.04051, 0.03756, 0.03883 | +5.41% |
| c32_ttfa_p50_ms | 180.6, 173.6, 176.2, 171.1 | 183.5, 189.4, 177.6, 180.4 | +4.20% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=6.2 cand=6.5
    iter_total: 30.060 -> 30.020
    worker.postprocess.event_sync: 13.975 -> 13.768
    worker.gpu_thread.prepare_inputs: 0.295 -> 0.305
    worker.gpu_thread.exec: 22.453 -> 24.903
    await_gpu: 15.745 -> 17.352
    worker.postprocess_batch: 15.594 -> 15.184
    worker.schedule: 0.018 -> 0.013

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.009411764705882352, "wer_cand_n": 32, "same_length_frac": 1.0, "waveform_snr_db_median": 19.80185317993164}

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

## qwen3tts (GPU 1)  base=6e9b686b5 perf/overnight-shared dirty=0  cand=76e63021c perf/ovn-integ-cand-f dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 101.7, 101.7 | 104.1, 103.5 | +2.06% |
| c16_audio_s_total | 430, 431.4 | 432.5, 432.3 | +0.40% |
| c16_failed | 0, 0 | 0, 0 |  |
| c16_rtf_p50 | 0.1559, 0.1585 | 0.1537, 0.1524 | -2.67% |
| c16_ttfa_p50_ms | 67.63, 61.47 | 67.32, 65.86 | +3.15% |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.06051, 0.06055 | 0.0607, 0.06118 | +0.67% |
| c1_ttfa_p50_ms | 17.86, 19.1 | 17.89, 18.11 | -2.60% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=12.0 cand=12.1
    iter_total: 7.765 -> 7.550
    worker.postprocess.event_sync: 2.508 -> 2.336
    worker.gpu_thread.prepare_inputs: 1.405 -> 1.355
    worker.gpu_thread.exec: 2.035 -> 2.020
    await_gpu: 0.109 -> 0.085
    worker.postprocess_batch: 4.101 -> 4.026
    worker.schedule: 0.010 -> 0.010
    worker.plan_thread.pre_plan: 0.601 -> 0.701
    submit_spec: 2.079 -> 2.134

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.01647058823529412, "wer_cand_n": 32, "same_length_frac": 0.09375, "waveform_snr_db_median": -2.8620309829711914}

## zimage (GPU 0)  base=6e9b686b5 perf/overnight-shared dirty=0  cand=76e63021c perf/ovn-integ-cand-f dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.091, 1.111 | 1.108, 1.144 | +2.31% |
| failed | 0, 0 | 0, 0 |  |
| lat_mean_s | 0.8568, 0.8468 | 0.8432, 0.8554 | -0.30% |
| lat_min_s | 0.8497, 0.8375 | 0.8402, 0.8453 | -0.10% |
| lat_p50_s | 0.8538, 0.8457 | 0.8444, 0.8555 | +0.02% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.150 -> 70.134
    worker.postprocess.event_sync: 71.068 -> 70.544
    worker.gpu_thread.prepare_inputs: 0.225 -> 0.467
    worker.gpu_thread.exec: 3.266 -> 3.392
    await_gpu: 2.510 -> 2.686
    worker.postprocess_batch: 71.400 -> 71.042
    worker.schedule: 0.020 -> 0.030
    submit_spec: 0.489 -> 0.620

Correctness: {"psnr_base_vs_cand_r0_min": 27.087200578653473, "psnr_base_vs_cand_r0_median": 36.34006262777231, "psnr_base_r0_vs_r1_min": 32.89345968969223, "psnr_base_r0_vs_r1_median": 36.17361417349545, "psnr_cand_r0_vs_r1_min": 26.850206661361142, "psnr_cand_r0_vs_r1_median": 31.119496591359294}
