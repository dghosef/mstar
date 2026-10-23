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
