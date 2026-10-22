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
