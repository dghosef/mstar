# cosmos3_nano: Cosmos3-Nano text->video 832x480x189, 35 steps



Arms: M* PR = config; M* main = config; SGLang diffusion 0.5.22 = speed_resident; vllm-omni 0.30.0 = base
Competitor choice: SGLang 0.5.22 serves Cosmos3 natively (registry lists Nano/Edge); its video defaults offload the DiT/text encoder to CPU, so the resident 'speed' flags are set, guardrails off, and every sampling knob is sent explicitly (its Edge defaults are gs 5.0 / shift 3.0). vllm-omni 0.30.0 also serves both (second arm). diffusers' Cosmos3OmniPipeline is offline-only (not queued)

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## t2v: 189 frames 832x480 @24 fps, gs 6.0, shift 10.0

### frames/s

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 3.384 [3.382-3.387] n=2 | 2.907 [2.905-2.910] n=2 | 2.878 [2.875-2.881] n=2 |
| M* main | 3.353 [3.351-3.355] n=2 | 2.885 [2.882-2.887] n=2 | 2.856 [2.854-2.858] n=2 |
| SGLang diffusion 0.5.22 | 3.479 [3.477-3.481] n=2 | 3.491 [3.489-3.494] n=2 | 3.492 [3.490-3.494] n=2 |
| vllm-omni 0.30.0 | 3.448 [3.448-3.448] n=2 | 3.478 [3.476-3.479] n=2 | 3.484 [3.484-3.485] n=2 |

### req/s

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 0.018 [0.018-0.018] n=2 | 0.015 [0.015-0.015] n=2 | 0.015 [0.015-0.015] n=2 |
| M* main | 0.018 [0.018-0.018] n=2 | 0.015 [0.015-0.015] n=2 | 0.015 [0.015-0.015] n=2 |
| SGLang diffusion 0.5.22 | 0.018 [0.018-0.018] n=2 | 0.018 [0.018-0.018] n=2 | 0.018 [0.018-0.018] n=2 |
| vllm-omni 0.30.0 | 0.018 [0.018-0.018] n=2 | 0.018 [0.018-0.018] n=2 | 0.018 [0.018-0.018] n=2 |

### latency p50 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 55.668 [55.643-55.693] n=2 | 127.903 [127.842-127.963] n=2 | 197.647 [197.404-197.889] n=2 |
| M* main | 56.217 [56.163-56.271] n=2 | 128.689 [128.595-128.782] n=2 | 198.430 [197.940-198.920] n=2 |
| SGLang diffusion 0.5.22 | 54.157 [54.149-54.164] n=2 | 107.959 [107.784-108.135] n=2 | 216.036 [215.917-216.155] n=2 |
| vllm-omni 0.30.0 | 54.811 [54.806-54.816] n=2 | 108.364 [108.298-108.431] n=2 | 216.589 [216.522-216.656] n=2 |

### latency p99 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 55.694 [55.649-55.738] n=2 | 130.651 [130.518-130.784] n=2 | 377.013 [376.600-377.427] n=2 |
| M* main | 56.225 [56.177-56.272] n=2 | 131.659 [131.579-131.740] n=2 | 380.090 [378.803-381.376] n=2 |
| SGLang diffusion 0.5.22 | 54.199 [54.151-54.248] n=2 | 108.271 [108.211-108.331] n=2 | 216.373 [216.240-216.507] n=2 |
| vllm-omni 0.30.0 | 54.836 [54.826-54.846] n=2 | 108.893 [108.874-108.913] n=2 | 217.221 [217.220-217.223] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 55.647 [55.620-55.674] n=2 | 127.880 [127.823-127.937] n=2 | 197.622 [197.380-197.864] n=2 |
| M* main | 56.195 [56.138-56.251] n=2 | 128.664 [128.571-128.757] n=2 | 198.408 [197.919-198.897] n=2 |
| SGLang diffusion 0.5.22 | 0.003 [0.003-0.004] n=2 | 0.004 [0.004-0.004] n=2 | 0.006 [0.006-0.006] n=2 |
| vllm-omni 0.30.0 | 54.810 [54.804-54.815] n=2 | 108.363 [108.296-108.429] n=2 | 216.586 [216.520-216.652] n=2 |

M* PR frames/s relative to each arm:

- M* PR vs M* main: c=1 +0.9%, c=2 +0.8%, c=4 +0.8%
- M* PR vs SGLang diffusion 0.5.22: c=1 -2.7%, c=2 -16.7%, c=4 -17.6%
- M* PR vs vllm-omni 0.30.0: c=1 -1.8%, c=2 -16.4%, c=4 -17.4%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_pr | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 23.5107421875, "psnr_vs_ref_median": null, "parity": true} |
| mstar_main | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 23.0712890625, "psnr_vs_ref_median": 30.94811795302226, "parity": true} |
| mstar_pr_b | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 23.5107421875, "psnr_vs_ref_median": 99.0, "parity": true} |
| sglang__speed_resident | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 23.5626220703125, "psnr_vs_ref_median": 17.52972636975258, "parity": true} |
| sglang__speed_resident_compile | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 23.5595703125, "psnr_vs_ref_median": 17.488041853929968, "parity": true} |
| vllmomni__base | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 24.1424560546875, "psnr_vs_ref_median": 17.255246807559395, "parity": true} |

Profile: phases_t2v_c1.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

Profile: phases_t2v_c4.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

