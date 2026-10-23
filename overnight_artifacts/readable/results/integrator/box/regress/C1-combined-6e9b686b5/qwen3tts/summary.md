## qwen3tts (GPU 1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=6e9b686b5 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 100.4, 100.4 | 100.3, 102 | +0.72% |
| c16_audio_s_total | 433.5, 437.1 | 432.4, 432 | -0.72% |
| c16_failed | 0, 0 | 0, 0 |  |
| c16_rtf_p50 | 0.1575, 0.1547 | 0.157, 0.1561 | +0.29% |
| c16_ttfa_p50_ms | 75.72, 65.4 | 59.52, 64.42 | -12.18% |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.06225, 0.06076 | 0.06139, 0.06137 | -0.20% |
| c1_ttfa_p50_ms | 19.29, 18.54 | 18.39, 18.86 | -1.52% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=12.1 cand=11.3
    iter_total: 7.735 -> 7.308
    worker.postprocess.event_sync: 2.296 -> 2.530
    worker.gpu_thread.prepare_inputs: 1.380 -> 1.413
    worker.gpu_thread.exec: 2.205 -> 1.423
    await_gpu: 0.149 -> 0.128
    worker.postprocess_batch: 4.100 -> 3.915
    worker.schedule: 0.010 -> None
    worker.plan_thread.pre_plan: 0.699 -> 0.731
    submit_spec: 2.138 -> 2.448

Correctness: {"wer_base": 0.011764705882352941, "wer_base_n": 32, "wer_cand": 0.01411764705882353, "wer_cand_n": 32, "same_length_frac": 0.0625, "waveform_snr_db_median": -2.9027621746063232}
