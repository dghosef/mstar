# Regression suite C4-combined-387ce372c  (2026-10-21 10:31:00)

## kokoro (GPU 3)  base=55b0215a3 perf/overnight-shared dirty=0  cand=387ce372c perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_rtf_p50 | 0.006101, 0.006161, 0.006233, 0.00584 | 0.005937, 0.006042, 0.005729, 0.005982 | -2.65% |
| c1_ttfa_p50_ms | 29.89, 31.04, 30.63, 28.16 | 28.36, 29.53, 29, 28.82 | -3.35% |
| c32_audio_s_per_s | 765, 780.7, 768.2, 794.5 | 797.1, 680.3, 723.1, 797.5 | -3.55% |
| c32_audio_s_total | 756.4, 756.4, 756.4, 756.4 | 756.4, 756.4, 756.4, 756.4 | +0.00% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_rtf_p50 | 0.03742, 0.03672, 0.03625, 0.03561 | 0.03598, 0.04448, 0.03769, 0.0355 | +5.25% |
| c32_ttfa_p50_ms | 174.3, 174.3, 171.1, 166 | 172.1, 205.6, 179, 163.2 | +4.99% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=7.2 cand=5.5
    iter_total: 35.870 -> 35.490
    worker.postprocess.event_sync: 14.990 -> 14.850
    worker.gpu_thread.prepare_inputs: 0.340 -> 0.290
    worker.gpu_thread.exec: 23.170 -> 22.020
    await_gpu: 15.660 -> 16.180
    worker.postprocess_batch: 20.450 -> 20.480
    worker.schedule: 0.010 -> 0.020

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.009411764705882352, "wer_cand_n": 32, "same_length_frac": 1.0, "waveform_snr_db_median": 19.31316375732422}

## qwen3tts (GPU 1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=387ce372c perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 105.4, 104.6, 103.9 | 102.4, 104, 105.4 | -0.66% |
| c16_audio_s_total | 437.3, 428.9, 432.1 | 433.5, 428.7, 434.8 | -0.09% |
| c16_failed | 0, 0, 0 | 0, 0, 0 |  |
| c16_rtf_p50 | 0.1525, 0.1506, 0.1523 | 0.1537, 0.1545, 0.1497 | +0.55% |
| c16_ttfa_p50_ms | 66.85, 75.25, 74.14 | 63.98, 64.22, 64.01 | -11.11% |
| c1_failed | 0, 0, 0 | 0, 0, 0 |  |
| c1_rtf_p50 | 0.06157, 0.06104, 0.06115 | 0.06131, 0.06135, 0.06143 | +0.19% |
| c1_ttfa_p50_ms | 19.29, 17.18, 17.88 | 18.51, 18.73, 19.05 | +3.56% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=10.9 cand=10.8
    iter_total: 7.192 -> 7.287
    worker.postprocess.event_sync: 2.507 -> 2.465
    worker.gpu_thread.prepare_inputs: 1.378 -> 1.455
    worker.gpu_thread.exec: 1.400 -> 1.449
    await_gpu: 0.097 -> 0.143
    worker.postprocess_batch: 3.833 -> 3.868
    worker.plan_thread.pre_plan: 0.719 -> 0.782
    submit_spec: 2.513 -> 2.518

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.011764705882352941, "wer_cand_n": 32, "same_length_frac": 0.09375, "waveform_snr_db_median": -3.035172939300537}

## zimage (GPU 2)  base=55b0215a3 perf/overnight-shared dirty=0  cand=387ce372c perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.08, 1.075, 1.092 | 1.105, 1.094, 1.1 | +1.59% |
| failed | 0, 0, 0 | 0, 0, 0 |  |
| lat_mean_s | 0.8684, 0.8631, 0.8529 | 0.8696, 0.8548, 0.8534 | -0.25% |
| lat_min_s | 0.8624, 0.8497, 0.8486 | 0.8539, 0.8488, 0.8466 | -0.44% |
| lat_p50_s | 0.8679, 0.8671, 0.8535 | 0.8645, 0.8543, 0.8516 | -0.70% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.682 -> 70.668
    worker.postprocess.event_sync: 71.608 -> 71.528
    worker.gpu_thread.prepare_inputs: 0.244 -> 0.217
    worker.gpu_thread.exec: 3.433 -> 3.163
    await_gpu: 2.522 -> 2.556
    worker.postprocess_batch: 71.980 -> 71.930
    worker.schedule: 0.020 -> 0.025
    submit_spec: 0.481 -> 0.465

Correctness: {"psnr_base_vs_cand_r0_min": 26.383727251351065, "psnr_base_vs_cand_r0_median": 35.909482521478324, "psnr_base_r0_vs_r1_min": 31.493826303014096, "psnr_base_r0_vs_r1_median": 35.61318980899239, "psnr_cand_r0_vs_r1_min": 30.59605316596702, "psnr_cand_r0_vs_r1_median": 34.5067596577008}
