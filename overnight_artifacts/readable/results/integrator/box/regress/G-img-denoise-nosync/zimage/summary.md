## zimage (GPU 2)  base=6e9b686b5 perf/overnight-shared dirty=0  cand=24dc3b635 perf/ovn-integ-cand-g dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.085, 1.089, 1.079 | 1.096, 1.1, 1.085 | +0.83% |
| failed | 0, 0, 0 | 0, 0, 0 |  |
| lat_mean_s | 0.8665, 0.8553, 0.8527 | 0.8515, 0.8603, 0.8506 | -0.47% |
| lat_min_s | 0.847, 0.8495, 0.8467 | 0.8446, 0.8562, 0.845 | +0.10% |
| lat_p50_s | 0.8597, 0.8539, 0.8545 | 0.8512, 0.8575, 0.8517 | -0.30% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.620 -> 70.596
    worker.postprocess.event_sync: 71.498 -> 71.500
    worker.gpu_thread.prepare_inputs: 0.316 -> 0.224
    worker.gpu_thread.exec: 3.323 -> 3.236
    await_gpu: 2.520 -> 2.518
    worker.postprocess_batch: 71.840 -> 71.858
    worker.schedule: 0.020 -> 0.025
    submit_spec: 0.486 -> 0.486

Correctness: {"psnr_base_vs_cand_r0_min": 27.5004745835038, "psnr_base_vs_cand_r0_median": 34.67540898126134, "psnr_base_r0_vs_r1_min": 30.913554410199552, "psnr_base_r0_vs_r1_median": 33.31741188797207, "psnr_cand_r0_vs_r1_min": 26.414743704299525, "psnr_cand_r0_vs_r1_median": 34.96753718329612}
