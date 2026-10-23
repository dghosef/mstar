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
