# cosmos3_edge: Cosmos3-Edge text->video 832x480x121, 20 steps



Arms: M* PR = config; M* main = config; SGLang diffusion 0.5.22 = speed_resident; vllm-omni 0.30.0 = base
Competitor choice: SGLang 0.5.22 serves Cosmos3 natively (registry lists Nano/Edge); its video defaults offload the DiT/text encoder to CPU, so the resident 'speed' flags are set, guardrails off, and every sampling knob is sent explicitly (its Edge defaults are gs 5.0 / shift 3.0). vllm-omni 0.30.0 also serves both (second arm). diffusers' Cosmos3OmniPipeline is offline-only (not queued)

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## t2v: 121 frames 832x480 @24 fps, gs 6.0, shift 12.0

### frames/s

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 18.880 [18.872-18.889] n=2 | 15.794 [15.680-15.908] n=2 | 15.759 [15.643-15.874] n=2 |
| M* main | 18.038 [18.019-18.056] n=2 | 15.544 [15.491-15.596] n=2 | 15.147 [15.014-15.280] n=2 |
| SGLang diffusion 0.5.22 | 17.407 [17.362-17.453] n=2 | 17.853 [17.838-17.868] n=2 | 17.799 [17.777-17.822] n=2 |
| vllm-omni 0.30.0 | 17.230 [17.202-17.259] n=2 | 18.102 [18.099-18.106] n=2 | 18.236 [18.227-18.245] n=2 |

### req/s

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 0.156 [0.156-0.156] n=2 | 0.131 [0.130-0.131] n=2 | 0.130 [0.129-0.131] n=2 |
| M* main | 0.149 [0.149-0.149] n=2 | 0.128 [0.128-0.129] n=2 | 0.125 [0.124-0.126] n=2 |
| SGLang diffusion 0.5.22 | 0.144 [0.143-0.144] n=2 | 0.148 [0.147-0.148] n=2 | 0.147 [0.147-0.147] n=2 |
| vllm-omni 0.30.0 | 0.142 [0.142-0.143] n=2 | 0.150 [0.150-0.150] n=2 | 0.151 [0.151-0.151] n=2 |

### latency p50 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 6.258 [6.256-6.260] n=2 | 15.040 [14.928-15.152] n=2 | 29.434 [28.774-30.095] n=2 |
| M* main | 6.562 [6.558-6.566] n=2 | 14.847 [14.680-15.014] n=2 | 30.820 [29.898-31.741] n=2 |
| SGLang diffusion 0.5.22 | 6.762 [6.760-6.764] n=2 | 13.266 [13.213-13.319] n=2 | 26.675 [26.670-26.679] n=2 |
| vllm-omni 0.30.0 | 6.982 [6.967-6.998] n=2 | 13.074 [13.067-13.081] n=2 | 26.209 [26.208-26.210] n=2 |

### latency p99 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 6.269 [6.266-6.272] n=2 | 15.639 [15.421-15.856] n=2 | 31.032 [30.990-31.074] n=2 |
| M* main | 6.570 [6.558-6.581] n=2 | 15.753 [15.751-15.756] n=2 | 32.268 [32.134-32.402] n=2 |
| SGLang diffusion 0.5.22 | 6.812 [6.764-6.860] n=2 | 13.472 [13.426-13.518] n=2 | 26.982 [26.940-27.024] n=2 |
| vllm-omni 0.30.0 | 7.005 [6.999-7.010] n=2 | 13.564 [13.526-13.603] n=2 | 26.712 [26.677-26.746] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 6.236 [6.234-6.239] n=2 | 15.011 [14.901-15.121] n=2 | 29.379 [28.722-30.036] n=2 |
| M* main | 6.538 [6.532-6.544] n=2 | 14.818 [14.653-14.983] n=2 | 30.757 [29.829-31.685] n=2 |
| SGLang diffusion 0.5.22 | 0.003 [0.003-0.003] n=2 | 0.004 [0.004-0.004] n=2 | 0.006 [0.006-0.007] n=2 |
| vllm-omni 0.30.0 | 6.981 [6.965-6.996] n=2 | 13.070 [13.063-13.077] n=2 | 26.201 [26.200-26.202] n=2 |

M* PR frames/s relative to each arm:

- M* PR vs M* main: c=1 +4.7%, c=2 +1.6%, c=4 +4.0%
- M* PR vs SGLang diffusion 0.5.22: c=1 +8.5%, c=2 -11.5%, c=4 -11.5%
- M* PR vs vllm-omni 0.30.0: c=1 +9.6%, c=2 -12.8%, c=4 -13.6%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_pr | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 20.648193359375, "psnr_vs_ref_median": null, "parity": true} |
| mstar_main | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 20.1690673828125, "psnr_vs_ref_median": 24.323923815937455, "parity": true} |
| mstar_pr_b | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 20.648193359375, "psnr_vs_ref_median": 99.0, "parity": true} |
| sglang__speed_resident | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 22.369384765625, "psnr_vs_ref_median": 20.943406676863166, "parity": true} |
| sglang__speed_resident_compile | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 22.0367431640625, "psnr_vs_ref_median": 20.49090414708032, "parity": true} |
| vllmomni__base | PASS | t2v: {"n": 4, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 21.0357666015625, "psnr_vs_ref_median": 23.069479658590232, "parity": true} |

Profile: phases_t2v_c4.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

