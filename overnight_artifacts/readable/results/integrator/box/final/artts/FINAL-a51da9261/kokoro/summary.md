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
