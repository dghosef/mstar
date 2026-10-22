## qwen3tts (GPU 1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=55b0215a3 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 99.43, 89.54 | 91.16, 98.36 | +0.29% |
| c16_audio_s_total | 437.4, 425.7 | 426.3, 435.8 | -0.12% |
| c16_failed | 0, 0 | 0, 0 |  |
| c16_rtf_p50 | 0.1562, 0.1742 | 0.1769, 0.1578 | +1.27% |
| c16_ttfa_p50_ms | 72.62, 68.6 | 74.09, 66.13 | -0.71% |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.06448, 0.06503 | 0.06702, 0.06537 | +2.22% |
| c1_ttfa_p50_ms | 21.06, 19.97 | 20.81, 18.92 | -3.18% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=12.0 cand=10.9
    iter_total: 7.990 -> 7.430
    worker.postprocess.event_sync: 1.610 -> 2.427
    worker.gpu_thread.prepare_inputs: 1.340 -> 1.473
    worker.gpu_thread.exec: 2.300 -> 1.560
    await_gpu: 0.120 -> 0.067
    worker.postprocess_batch: 3.910 -> 4.000
    worker.schedule: 0.010 -> None
    worker.plan_thread.pre_plan: 0.860 -> 0.805
    submit_spec: 2.270 -> 2.583

Correctness: {"wer_base": 0.018823529411764704, "wer_base_n": 32, "wer_cand": 0.01411764705882353, "wer_cand_n": 32, "same_length_frac": 0.09375, "waveform_snr_db_median": -2.852879285812378}
