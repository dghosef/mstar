# Regression suite A-qwen35vl-one-row  (2026-10-21 06:57:11)

## qwen35 (GPU 0)  base=55b0215a3 perf/overnight-shared dirty=0  cand=2c41f4a64 perf/ovn-integ-cand dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.607, 1.382, 1.45, 1.388 | 1.398, 1.381, 1.522, 1.39 | -2.31% |
| c1_tok_s | 605.5, 704.6, 670.6, 701.1 | 695.4, 705.3, 639.7, 700.5 | +2.20% |
| c1_ttft_ms | 12.38, 10.69, 11.4, 11.23 | 11.54, 10.79, 11.64, 10.83 | -1.95% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 2.352, 2.174, 2.257, 2.141 | 2.181, 2.073, 2.115, 2.112 | -4.96% |
| c32_tok_s | 1.27e+04, 1.362e+04, 1.314e+04, 1.391e+04 | 1.36e+04, 1.431e+04, 1.4e+04, 1.404e+04 | +4.81% |
| c32_ttft_p50_ms | 42.58, 45.12, 46.98, 40.46 | 46.59, 41.27, 43, 38.99 | -3.02% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.683, 3.623, 3.673, 3.738 | 3.928, 3.611, 3.706, 3.684 | +1.44% |
| c64_tok_s | 1.596e+04, 1.628e+04, 1.588e+04, 1.56e+04 | 1.489e+04, 1.628e+04, 1.576e+04, 1.596e+04 | -1.29% |
| c64_ttft_p50_ms | 84, 79.37, 94.92, 87.87 | 91.94, 81.85, 90.24, 84.91 | +0.80% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=61.2 cand=61.5
    iter_total: 2.978 -> 2.979
    worker.postprocess.event_sync: 0.101 -> 0.100
    worker.gpu_thread.prepare_inputs: 0.024 -> 0.024
    worker.gpu_thread.exec: 1.552 -> 1.585
    await_gpu: 0.660 -> 0.671
    worker.postprocess_batch: 0.566 -> 0.588
    worker.schedule: 0.021 -> 0.010
    worker.plan_thread.pre_plan: 1.100 -> 1.076
    submit_spec: 1.108 -> 1.066

Correctness: {"base_vs_cand_match": 4, "n": 8, "base_r0_vs_r1_match": 4, "cand_r0_vs_r1_match": 8, "cand_outputs_seen_in_base": 8, "first_diff": {"prompt": 2, "common_prefix_chars": 363, "base": ".\n    *   The sky appears blue because of this scattering.\n\n3.  **Drafting Sente", "cand": " efficiently.\n    *   The sky appears blue because of this scattering.\n\n3.  **Dr"}}

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
