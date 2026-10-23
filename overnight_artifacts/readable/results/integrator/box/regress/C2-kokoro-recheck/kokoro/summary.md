## kokoro (GPU 0)  base=55b0215a3 perf/overnight-shared dirty=0  cand=6e9b686b5 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_rtf_p50 | 0.005899, 0.005926, 0.006974, 0.005788 | 0.005753, 0.006044, 0.005789, 0.005742 | -5.12% |
| c1_ttfa_p50_ms | 28.29, 28.53, 32.82, 28.4 | 27.46, 29.68, 27.8, 27.43 | -4.81% |
| c32_audio_s_per_s | 782.9, 773.8, 655, 704.1 | 767.4, 715.5, 776.2, 813.2 | +5.37% |
| c32_audio_s_total | 756.4, 756.4, 756.4, 756.4 | 756.5, 756.4, 756.4, 756.4 | +0.00% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_rtf_p50 | 0.03593, 0.03688, 0.04122, 0.04083 | 0.03731, 0.03942, 0.0364, 0.03579 | -3.83% |
| c32_ttfa_p50_ms | 171.7, 170.3, 194.6, 190.2 | 178.6, 184.4, 173.3, 166.7 | -3.26% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=5.3 cand=6.2
    iter_total: 33.480 -> 33.025
    worker.postprocess.event_sync: 13.710 -> 12.525
    worker.gpu_thread.prepare_inputs: 0.290 -> 0.295
    worker.gpu_thread.exec: 21.020 -> 22.746
    await_gpu: 16.960 -> 14.519
    worker.postprocess_batch: 18.190 -> 13.964
    worker.schedule: 0.010 -> 0.013

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.009411764705882352, "wer_cand_n": 32, "same_length_frac": 1.0, "waveform_snr_db_median": 19.618600845336914}
