## kokoro (GPU 0)  base=?  cand=?
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |

Profile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):
- worker_0: bs base=8.5 cand=9.0
    iter_total: 34.183 -> 33.917
    worker.postprocess.event_sync: 15.630 -> 14.705
    worker.gpu_thread.prepare_inputs: 0.343 -> 0.338
    worker.gpu_thread.exec: 30.197 -> 29.300
    await_gpu: 22.423 -> 23.288
    worker.postprocess_batch: 17.393 -> 16.095
    worker.schedule: 0.090 -> 0.094

Correctness: {"wer_base": null, "wer_base_n": 0, "wer_cand": null, "wer_cand_n": 0, "same_length_frac": 0.0, "waveform_snr_db_median": null}
