## zimage (GPU 3)  base=55b0215a3 perf/overnight-shared dirty=0  cand=b53845708 perf/ovn-integ-cand-b dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.13, 1.092 | 1.074, 1.09 | -2.59% |
| failed | 0, 0 | 0, 0 |  |
| lat_mean_s | 0.8601, 0.8588 | 0.8687, 0.855 | +0.28% |
| lat_min_s | 0.8496, 0.8505 | 0.8579, 0.8484 | +0.36% |
| lat_p50_s | 0.8585, 0.8567 | 0.864, 0.8544 | +0.19% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.558 -> 70.594
    worker.postprocess.event_sync: 71.350 -> 71.264
    worker.gpu_thread.prepare_inputs: 0.308 -> 0.261
    worker.gpu_thread.exec: 3.158 -> 3.302
    await_gpu: 2.406 -> 2.554
    worker.postprocess_batch: 71.744 -> 71.834
    worker.schedule: 0.030 -> 0.030
    submit_spec: 0.545 -> 0.572

Correctness: {"psnr_base_vs_cand_r0_min": 30.34426668121752, "psnr_base_vs_cand_r0_median": 35.50585759032445, "psnr_base_r0_vs_r1_min": 28.309505069203937, "psnr_base_r0_vs_r1_median": 33.333071615898554, "psnr_cand_r0_vs_r1_min": 31.51272900530323, "psnr_cand_r0_vs_r1_median": 36.207936950022344}
