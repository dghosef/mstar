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
