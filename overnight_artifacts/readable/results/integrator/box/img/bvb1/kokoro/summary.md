## kokoro (GPU 1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=55b0215a3 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.006926, 0.006289 | 0.009924, 0.007092 | +28.76% |
| c1_ttfa_p50_ms | 31.94, 31.07 | 44.15, 36.39 | +27.82% |
| c32_audio_s_per_s | 580, 648.9 | 470.2, 652.9 | -8.62% |
| c32_audio_s_total | 756.4, 756.4 | 756.4, 756.4 | -0.00% |
| c32_failed | 0, 0 | 0, 0 |  |
| c32_rtf_p50 | 0.04376, 0.04368 | 0.04994, 0.04607 | +9.80% |
| c32_ttfa_p50_ms | 209.6, 209.3 | 233.3, 221.3 | +8.53% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.009411764705882352, "wer_cand_n": 32, "same_length_frac": 0.96875, "waveform_snr_db_median": 19.2109375}
