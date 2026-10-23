## zimage (GPU 2)  base=55b0215a3 perf/overnight-shared dirty=0  cand=6e9b686b5 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.078, 1.079 | 1.078, 1.092 | +0.64% |
| failed | 0, 0 | 0, 0 |  |
| lat_mean_s | 0.8587, 0.8549 | 0.8619, 0.8554 | +0.22% |
| lat_min_s | 0.8518, 0.8496 | 0.8545, 0.85 | +0.18% |
| lat_p50_s | 0.8561, 0.8546 | 0.8609, 0.8532 | +0.20% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.744 -> 71.006
    worker.postprocess.event_sync: 71.582 -> 71.810
    worker.gpu_thread.prepare_inputs: 0.231 -> 0.418
    worker.gpu_thread.exec: 3.238 -> 3.146
    await_gpu: 2.534 -> 2.414
    worker.postprocess_batch: 72.000 -> 72.122
    worker.schedule: 0.030 -> 0.020
    submit_spec: 0.494 -> 0.504

Correctness: {"psnr_base_vs_cand_r0_min": 31.739912812352035, "psnr_base_vs_cand_r0_median": 35.861414799167875, "psnr_base_r0_vs_r1_min": 31.08090778680339, "psnr_base_r0_vs_r1_median": 37.621791898753095, "psnr_cand_r0_vs_r1_min": 29.751755917346575, "psnr_cand_r0_vs_r1_median": 33.96804953990241}
