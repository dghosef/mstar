## zimage (GPU 2)  base=55b0215a3 perf/overnight-shared dirty=0  cand=387ce372c perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.08, 1.075, 1.092 | 1.105, 1.094, 1.1 | +1.59% |
| failed | 0, 0, 0 | 0, 0, 0 |  |
| lat_mean_s | 0.8684, 0.8631, 0.8529 | 0.8696, 0.8548, 0.8534 | -0.25% |
| lat_min_s | 0.8624, 0.8497, 0.8486 | 0.8539, 0.8488, 0.8466 | -0.44% |
| lat_p50_s | 0.8679, 0.8671, 0.8535 | 0.8645, 0.8543, 0.8516 | -0.70% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.682 -> 70.668
    worker.postprocess.event_sync: 71.608 -> 71.528
    worker.gpu_thread.prepare_inputs: 0.244 -> 0.217
    worker.gpu_thread.exec: 3.433 -> 3.163
    await_gpu: 2.522 -> 2.556
    worker.postprocess_batch: 71.980 -> 71.930
    worker.schedule: 0.020 -> 0.025
    submit_spec: 0.481 -> 0.465

Correctness: {"psnr_base_vs_cand_r0_min": 26.383727251351065, "psnr_base_vs_cand_r0_median": 35.909482521478324, "psnr_base_r0_vs_r1_min": 31.493826303014096, "psnr_base_r0_vs_r1_median": 35.61318980899239, "psnr_cand_r0_vs_r1_min": 30.59605316596702, "psnr_cand_r0_vs_r1_median": 34.5067596577008}
