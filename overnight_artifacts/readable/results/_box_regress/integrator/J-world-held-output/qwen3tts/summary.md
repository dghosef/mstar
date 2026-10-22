## qwen3tts (GPU 1)  base=d1c91505a perf/overnight-shared dirty=0  cand=bec51e1c5 perf/ovn-integ-cand-j dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c16_audio_s_per_s | 97.09, 97.31, 93.55 | 82.22, 97.62, 102.5 | -1.94% |
| c16_audio_s_total | 424.5, 432.7, 434.9 | 427.2, 427.9, 428.4 | -0.66% |
| c16_failed | 0, 0, 0 | 0, 0, 0 |  |
| c16_rtf_p50 | 0.1625, 0.1633, 0.1654 | 0.1895, 0.1583, 0.1541 | +2.19% |
| c16_ttfa_p50_ms | 77.17, 69.38, 62.81 | 90.04, 77.74, 62.68 | +10.08% |
| c1_failed | 0, 0, 0 | 0, 0, 0 |  |
| c1_rtf_p50 | 0.06371, 0.06263, 0.06142 | 0.06518, 0.06109, 0.06179 | +0.16% |
| c1_ttfa_p50_ms | 20.86, 20.77, 20.16 | 23.51, 20, 17.69 | -0.98% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=10.1 cand=10.6
    iter_total: 7.582 -> 7.453
    worker.postprocess.event_sync: 2.425 -> 2.470
    worker.gpu_thread.prepare_inputs: 1.505 -> 1.433
    worker.gpu_thread.exec: 1.610 -> 1.543
    await_gpu: 0.105 -> 0.097
    worker.postprocess_batch: 3.833 -> 3.923
    worker.plan_thread.pre_plan: 0.843 -> 0.860
    submit_spec: 2.800 -> 2.600

Correctness: {"wer_base": 0.007058823529411765, "wer_base_n": 32, "wer_cand": 0.018823529411764704, "wer_cand_n": 32, "same_length_frac": 0.0625, "waveform_snr_db_median": -3.1382830142974854}
