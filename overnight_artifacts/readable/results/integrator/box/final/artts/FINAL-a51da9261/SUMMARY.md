# Regression suite FINAL-a51da9261  (2026-10-21 20:11:55)

## kokoro (GPU 0)  base=55b0215a3 HEAD dirty=0  cand=a51da9261 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_rtf_p50 | 0.006122, 0.005926, 0.006481, 0.006195 | 0.005911, 0.006143, 0.005835, 0.005733 | -4.46% |
| c1_ttfa_p50_ms | 30.28, 28.69, 30.95, 29.57 | 28.67, 29.34, 27.74, 27.37 | -5.32% |
| c32_audio_s_per_s | 825.6, 754.5, 482.2, 746.3 | 718, 748.3, 739, 725.8 | +4.36% |
| c32_audio_s_total | 756.4, 756.4, 756.4, 756.4 | 756.4, 756.4, 756.4, 756.4 | +0.00% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_rtf_p50 | 0.03572, 0.03786, 0.05837, 0.03933 | 0.04057, 0.03868, 0.04021, 0.03948 | -7.20% |
| c32_ttfa_p50_ms | 168.6, 177.5, 268.7, 184.7 | 191.2, 182.2, 189.4, 187.4 | -6.16% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=5.1 cand=6.7
    iter_total: 30.770 -> 30.760
    worker.postprocess.event_sync: 10.890 -> 14.050
    worker.gpu_thread.prepare_inputs: 0.250 -> 0.290
    worker.gpu_thread.exec: 21.520 -> 22.820
    await_gpu: 14.160 -> 15.250
    worker.postprocess_batch: 12.170 -> 15.630
    worker.schedule: 0.010 -> 0.020

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.009411764705882352, "wer_cand_n": 32, "same_length_frac": 0.96875, "waveform_snr_db_median": 18.979896545410156}

## qwen3tts (GPU 0)  base=55b0215a3 HEAD dirty=0  cand=a51da9261 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 95.81, 45.83, 83.03, 98.98 | 91.85, 88.55, 103.2, 46.14 | +1.88% |
| c16_audio_s_total | 437.1, 435.5, 433, 435.3 | 430.7, 428.7, 436.6, 427.1 | -1.02% |
| c16_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c16_rtf_p50 | 0.1639, 0.3485, 0.1895, 0.1565 | 0.1696, 0.1814, 0.1508, 0.3459 | -1.24% |
| c16_ttfa_p50_ms | 71.35, 142.4, 73.49, 70.41 | 65.53, 65.32, 64.42, 145.6 | -4.70% |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_rtf_p50 | 0.06195, 0.1312, 0.07034, 0.06246 | 0.07637, 0.06241, 0.06245, 0.1329 | +2.52% |
| c1_ttfa_p50_ms | 18.8, 29.1, 21.58, 18.87 | 21.03, 17.76, 19.03, 30.3 | -0.26% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=10.8 cand=10.3
    iter_total: 7.213 -> 6.841
    worker.postprocess.event_sync: 2.473 -> 2.331
    worker.gpu_thread.prepare_inputs: 1.453 -> 1.391
    worker.gpu_thread.exec: 1.353 -> 1.398
    await_gpu: 0.065 -> 0.106
    worker.postprocess_batch: 3.812 -> 3.587
    worker.schedule: None -> 0.040
    worker.plan_thread.pre_plan: 0.797 -> 0.696
    submit_spec: 2.555 -> 2.464

Correctness: {"wer_base": 0.007058823529411765, "wer_base_n": 32, "wer_cand": 0.01647058823529412, "wer_cand_n": 32, "same_length_frac": 0.15625, "waveform_snr_db_median": -2.5779056549072266}
