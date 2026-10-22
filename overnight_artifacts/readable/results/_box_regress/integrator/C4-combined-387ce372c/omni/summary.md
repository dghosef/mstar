## omni (GPU 0,1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=387ce372c perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.07767, 0.08369 | 0.08208, 0.08112 | +1.14% |
| c1_ttfa_p50_ms | 25.7, 26.84 | 27.5, 25.12 | +0.14% |
| c8_audio_s_per_s | 51.73, 50.81 | 51.62, 51.51 | +0.58% |
| c8_audio_s_total | 127.1, 128.7 | 128.4, 128.7 | +0.53% |
| c8_failed | 0, 0 | 0, 0 |  |
| c8_rtf_p50 | 0.1482, 0.1487 | 0.1523, 0.1513 | +2.28% |
| c8_ttfa_p50_ms | 50.08, 58.52 | 48.02, 43.05 | -16.14% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=4.9 cand=7.2
    iter_total: 6.763 -> 8.950
    worker.postprocess.event_sync: 1.993 -> 2.620
    worker.gpu_thread.prepare_inputs: 1.150 -> 1.200
    worker.gpu_thread.exec: 1.617 -> 2.140
    await_gpu: 0.250 -> 0.040
    worker.postprocess_batch: 3.150 -> 4.010
    worker.schedule: 0.033 -> 0.000
    worker.plan_thread.pre_plan: 0.810 -> 0.660
    submit_spec: 2.211 -> 1.720
- worker_1: bs base=0.0 cand=3.4
    iter_total: None -> 9.787
    worker.postprocess.event_sync: None -> 0.080
    worker.gpu_thread.prepare_inputs: None -> 0.367
    worker.gpu_thread.exec: None -> 9.310
    await_gpu: None -> 6.209
    worker.postprocess_batch: None -> 1.758
    worker.schedule: None -> 0.011
    worker.plan_thread.pre_plan: None -> 0.538
    submit_spec: None -> 1.221

Correctness: {"wer_base": 6.5717647058823525, "wer_base_n": 32, "wer_cand": 7.305882352941176, "wer_cand_n": 32, "same_length_frac": 0.1875, "waveform_snr_db_median": -2.999239206314087}
