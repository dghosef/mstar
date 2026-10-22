## kokoro (GPU 0)  base=?  cand=?
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_rtf_p50 | 0.005738, 0.005807, 0.005952, 0.006098 | 0.006058, 0.005843, 0.006029, 0.006352 | +2.92% |
| c1_ttfa_p50_ms | 27.47, 27.89, 28.73, 29.14 | 28.97, 27.9, 28.76, 30.01 | +2.12% |
| c32_audio_s_per_s | 793.3, 786.5, 685.3, 759.9 | 764.5, 766.2, 764.2, 786.5 | +1.86% |
| c32_audio_s_total | 756.4, 756.4, 756.4, 756.4 | 756.4, 756.4, 756.4, 756.4 | +0.00% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_rtf_p50 | 0.03579, 0.03682, 0.04184, 0.03995 | 0.0372, 0.03745, 0.0373, 0.03671 | -3.71% |
| c32_ttfa_p50_ms | 167.8, 176.5, 198.6, 187.7 | 179.3, 181.8, 177.2, 176.8 | -2.12% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=5.3 cand=7.2
    iter_total: 35.870 -> 36.420
    worker.postprocess.event_sync: 13.640 -> 16.790
    worker.gpu_thread.prepare_inputs: 0.290 -> 0.320
    worker.gpu_thread.exec: 23.240 -> 26.090
    await_gpu: 16.110 -> 17.000
    worker.postprocess_batch: 16.820 -> 18.340
    worker.schedule: 0.020 -> 0.030

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.009411764705882352, "wer_cand_n": 32, "same_length_frac": 1.0, "waveform_snr_db_median": 19.45132064819336}
