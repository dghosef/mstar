## zimage (GPU 1)  base=55b0215a3 HEAD dirty=0  cand=a51da9261 perf/overnight-shared dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c4_img_per_s | 1.128, 1.086, 1.087, 1.096 | 1.115, 1.103, 1.136, 1.098 | +1.25% |
| failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| lat_mean_s | 0.8583, 0.8504, 0.8531, 0.8572 | 0.8533, 0.8524, 0.849, 0.8537 | -0.31% |
| lat_min_s | 0.8473, 0.8445, 0.8438, 0.8471 | 0.8444, 0.8495, 0.8373, 0.8451 | -0.19% |
| lat_p50_s | 0.8571, 0.8491, 0.8504, 0.8591 | 0.8522, 0.8516, 0.8501, 0.8553 | -0.19% |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=1.0 cand=1.0
    iter_total: 70.366 -> 70.318
    worker.postprocess.event_sync: 71.240 -> 70.884
    worker.gpu_thread.prepare_inputs: 0.231 -> 0.274
    worker.gpu_thread.exec: 3.312 -> 3.129
    await_gpu: 2.476 -> 2.422
    worker.postprocess_batch: 71.666 -> 71.412
    worker.schedule: 0.030 -> 0.030
    submit_spec: 0.484 -> 0.590

Correctness: {"psnr_base_vs_cand_r0_min": 32.30935983245371, "psnr_base_vs_cand_r0_median": 34.61973779938907, "psnr_base_r0_vs_r1_min": 32.24601555098156, "psnr_base_r0_vs_r1_median": 36.285502682794984, "psnr_cand_r0_vs_r1_min": 32.39115016474034, "psnr_cand_r0_vs_r1_median": 34.379795999238894}
