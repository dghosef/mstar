## omni (GPU 2,3)  base=d1c91505a perf/overnight-shared dirty=0  cand=bec51e1c5 perf/ovn-integ-cand-j dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_rtf_p50 | 0.08234, 0.08191 | 0.08303, 0.08398 | +1.69% |
| c1_ttfa_p50_ms | 27.74, 27.56 | 25.22, 27.34 | -4.94% |
| c8_audio_s_per_s | 51.78, 51.4 | 51.71, 51.08 | -0.38% |
| c8_audio_s_total | 124.8, 129.5 | 127.9, 122.7 | -1.45% |
| c8_failed | 0, 0 | 0, 0 |  |
| c8_rtf_p50 | 0.1502, 0.152 | 0.1478, 0.1488 | -1.82% |
| c8_ttfa_p50_ms | 57.04, 49.83 | 46.45, 45.39 | -14.06% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=7.2 cand=7.2
    iter_total: 8.900 -> 8.960
    worker.postprocess.event_sync: 2.580 -> 2.430
    worker.gpu_thread.prepare_inputs: 1.280 -> 1.250
    worker.gpu_thread.exec: 2.030 -> 1.940
    await_gpu: 0.000 -> 0.030
    worker.postprocess_batch: 3.920 -> 4.190
    worker.schedule: 0.000 -> 0.000
    worker.plan_thread.pre_plan: 0.700 -> 0.740
    submit_spec: 1.800 -> 1.800
- worker_1: bs base=3.2 cand=3.5
    iter_total: 8.257 -> 9.010
    worker.postprocess.event_sync: 0.037 -> 0.030
    worker.gpu_thread.prepare_inputs: 0.300 -> 0.320
    worker.gpu_thread.exec: 8.136 -> 8.714
    await_gpu: 5.401 -> 5.928
    worker.postprocess_batch: 1.546 -> 1.634
    worker.schedule: 0.014 -> 0.011
    worker.plan_thread.pre_plan: 0.700 -> 0.664
    submit_spec: 1.110 -> 1.099

ERRORS: ["correctness: ValueError('buffer size must be a multiple of element size')"]
