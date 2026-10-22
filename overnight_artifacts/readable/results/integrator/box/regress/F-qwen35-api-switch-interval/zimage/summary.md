## zimage (GPU 0)  base=6e9b686b5 perf/overnight-shared dirty=0  cand=76e63021c perf/ovn-integ-cand-f dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.091, 1.111 | 1.108, 1.144 | +2.31% |
| failed | 0, 0 | 0, 0 |  |
| lat_mean_s | 0.8568, 0.8468 | 0.8432, 0.8554 | -0.30% |
| lat_min_s | 0.8497, 0.8375 | 0.8402, 0.8453 | -0.10% |
| lat_p50_s | 0.8538, 0.8457 | 0.8444, 0.8555 | +0.02% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.150 -> 70.134
    worker.postprocess.event_sync: 71.068 -> 70.544
    worker.gpu_thread.prepare_inputs: 0.225 -> 0.467
    worker.gpu_thread.exec: 3.266 -> 3.392
    await_gpu: 2.510 -> 2.686
    worker.postprocess_batch: 71.400 -> 71.042
    worker.schedule: 0.020 -> 0.030
    submit_spec: 0.489 -> 0.620

Correctness: {"psnr_base_vs_cand_r0_min": 27.087200578653473, "psnr_base_vs_cand_r0_median": 36.34006262777231, "psnr_base_r0_vs_r1_min": 32.89345968969223, "psnr_base_r0_vs_r1_median": 36.17361417349545, "psnr_cand_r0_vs_r1_min": 26.850206661361142, "psnr_cand_r0_vs_r1_median": 31.119496591359294}
