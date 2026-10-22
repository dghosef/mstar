## qwen3tts (GPU 1)  base=55b0215a3 perf/overnight-shared dirty=0  cand=2c41f4a64 perf/ovn-integ-cand dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 102.4, 101.4 | 102.6, 103.1 | +0.95% |
| c16_audio_s_total | 431.4, 429.4 | 433.8, 429.8 | +0.33% |
| c16_failed | 0, 0 | 0, 0 |  |
| c16_rtf_p50 | 0.1542, 0.1552 | 0.1536, 0.154 | -0.58% |
| c16_ttfa_p50_ms | 64.3, 59.74 | 71.78, 69.57 | +13.96% |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.06163, 0.06401 | 0.06165, 0.0608 | -2.54% |
| c1_ttfa_p50_ms | 19.24, 20.98 | 20.43, 18.8 | -2.46% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=11.0 cand=11.2
    iter_total: 7.290 -> 7.335
    worker.postprocess.event_sync: 2.340 -> 2.593
    worker.gpu_thread.prepare_inputs: 1.425 -> 1.423
    worker.gpu_thread.exec: 1.527 -> 1.545
    await_gpu: 0.107 -> 0.152
    worker.postprocess_batch: 3.897 -> 3.972
    worker.plan_thread.pre_plan: 0.765 -> 0.691
    submit_spec: 2.528 -> 2.467

Correctness: {"wer_base": 0.009411764705882352, "wer_base_n": 32, "wer_cand": 0.004705882352941176, "wer_cand_n": 32, "same_length_frac": 0.09375, "waveform_snr_db_median": -2.9820733070373535}
