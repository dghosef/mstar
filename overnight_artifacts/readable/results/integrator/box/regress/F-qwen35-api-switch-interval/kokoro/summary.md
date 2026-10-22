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
