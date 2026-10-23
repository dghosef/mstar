## qwen3tts (GPU 1)  base=6e9b686b5 perf/overnight-shared dirty=0  cand=76e63021c perf/ovn-integ-cand-f dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 101.7, 101.7 | 104.1, 103.5 | +2.06% |
| c16_audio_s_total | 430, 431.4 | 432.5, 432.3 | +0.40% |
| c16_failed | 0, 0 | 0, 0 |  |
| c16_rtf_p50 | 0.1559, 0.1585 | 0.1537, 0.1524 | -2.67% |
| c16_ttfa_p50_ms | 67.63, 61.47 | 67.32, 65.86 | +3.15% |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.06051, 0.06055 | 0.0607, 0.06118 | +0.67% |
| c1_ttfa_p50_ms | 17.86, 19.1 | 17.89, 18.11 | -2.60% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=12.0 cand=12.1
    iter_total: 7.765 -> 7.550
    worker.postprocess.event_sync: 2.508 -> 2.336
    worker.gpu_thread.prepare_inputs: 1.405 -> 1.355
    worker.gpu_thread.exec: 2.035 -> 2.020
    await_gpu: 0.109 -> 0.085
    worker.postprocess_batch: 4.101 -> 4.026
    worker.schedule: 0.010 -> 0.010
    worker.plan_thread.pre_plan: 0.601 -> 0.701
    submit_spec: 2.079 -> 2.134

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.01647058823529412, "wer_cand_n": 32, "same_length_frac": 0.09375, "waveform_snr_db_median": -2.8620309829711914}
