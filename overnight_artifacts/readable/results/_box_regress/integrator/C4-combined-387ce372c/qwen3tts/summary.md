## qwen3tts (GPU 1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=387ce372c perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 105.4, 104.6, 103.9 | 102.4, 104, 105.4 | -0.66% |
| c16_audio_s_total | 437.3, 428.9, 432.1 | 433.5, 428.7, 434.8 | -0.09% |
| c16_failed | 0, 0, 0 | 0, 0, 0 |  |
| c16_rtf_p50 | 0.1525, 0.1506, 0.1523 | 0.1537, 0.1545, 0.1497 | +0.55% |
| c16_ttfa_p50_ms | 66.85, 75.25, 74.14 | 63.98, 64.22, 64.01 | -11.11% |
| c1_failed | 0, 0, 0 | 0, 0, 0 |  |
| c1_rtf_p50 | 0.06157, 0.06104, 0.06115 | 0.06131, 0.06135, 0.06143 | +0.19% |
| c1_ttfa_p50_ms | 19.29, 17.18, 17.88 | 18.51, 18.73, 19.05 | +3.56% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=10.9 cand=10.8
    iter_total: 7.192 -> 7.287
    worker.postprocess.event_sync: 2.507 -> 2.465
    worker.gpu_thread.prepare_inputs: 1.378 -> 1.455
    worker.gpu_thread.exec: 1.400 -> 1.449
    await_gpu: 0.097 -> 0.143
    worker.postprocess_batch: 3.833 -> 3.868
    worker.plan_thread.pre_plan: 0.719 -> 0.782
    submit_spec: 2.513 -> 2.518

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.011764705882352941, "wer_cand_n": 32, "same_length_frac": 0.09375, "waveform_snr_db_median": -3.035172939300537}
