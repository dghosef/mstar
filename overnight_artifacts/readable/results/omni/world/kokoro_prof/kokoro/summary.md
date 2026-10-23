## kokoro (GPU 0)  base=?  cand=?
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0 | 0 |  |
| c1_rtf_p50 | 0.00528 | 0.005555 | +5.20% |
| c1_ttfa_p50_ms | 25.25 | 26.41 | +4.58% |
| c32_audio_s_per_s | 786.1 | 781.7 | -0.56% |
| c32_audio_s_total | 756.4 | 756.4 | -0.00% |
| c32_failed | 0 | 0 |  |
| c32_rtf_p50 | 0.03735 | 0.03775 | +1.07% |
| c32_ttfa_p50_ms | 175.3 | 180.5 | +2.93% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=7.2 cand=5.4
    iter_total: 28.850 -> 33.360
    worker.postprocess.event_sync: 13.570 -> 13.720
    worker.gpu_thread.prepare_inputs: 0.280 -> 0.280
    worker.gpu_thread.exec: 26.460 -> 21.100
    await_gpu: 19.170 -> 13.900
    worker.postprocess_batch: 15.440 -> 15.430
    worker.schedule: 0.010 -> 0.010

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.009411764705882352, "wer_cand_n": 32, "same_length_frac": 1.0, "waveform_snr_db_median": 19.64649200439453}
