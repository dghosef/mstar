# bagel_2gpu: BAGEL-7B-MoT on 2 GPUs (CFG parallel)



Arms: M* PR = config; M* main = config; vllm-omni 0.30.0 = cfgp2
Competitor choice: vllm-omni 0.30.0 serves BAGEL t2i/edit (since v0.12.0rc1) through /v1/chat/completions with every CFG knob sent explicitly (its /v1/images routes drop them; its defaults differ: cfg_img 1.5, shift 1.0 on one path); vLLM 0.30.0 core serves understanding (streams, so it is the stronger understanding arm). SGLang/sglang-omni/diffusers have no BAGEL. The official repo is serial and pinned to py3.10/torch 2.5 (not queued). M*'s cfg_prune runs the same CFG branches as the reference (41 dB vs base, inside the 40 dB same-code floor)

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## t2i: text -> image 1024x1024, 50 steps, cfg_text 4.0 (BAGEL defaults), PartiPrompts

### img/s

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 0.167 [0.167-0.167] n=2 | 0.167 [0.167-0.167] n=2 | 0.171 [0.171-0.171] n=2 |
| M* main | 0.085 [0.085-0.085] n=2 | 0.087 [0.087-0.087] n=2 | 0.088 [0.088-0.088] n=2 |
| vllm-omni 0.30.0 | 0.140 [0.139-0.140] n=2 | 0.146 [0.146-0.146] n=2 | 0.146 [0.146-0.146] n=2 |

### req/s

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 0.167 [0.167-0.167] n=2 | 0.167 [0.167-0.167] n=2 | 0.171 [0.171-0.171] n=2 |
| M* main | 0.085 [0.085-0.085] n=2 | 0.087 [0.087-0.087] n=2 | 0.088 [0.088-0.088] n=2 |
| vllm-omni 0.30.0 | 0.140 [0.139-0.140] n=2 | 0.146 [0.146-0.146] n=2 | 0.146 [0.146-0.146] n=2 |

### latency p50 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 5.965 [5.955-5.975] n=2 | 11.865 [11.850-11.881] n=2 | 23.174 [23.152-23.197] n=2 |
| M* main | 11.745 [11.737-11.752] n=2 | 22.865 [22.812-22.918] n=2 | 45.186 [45.148-45.223] n=2 |
| vllm-omni 0.30.0 | 7.127 [7.125-7.128] n=2 | 13.551 [13.538-13.564] n=2 | 27.208 [27.183-27.234] n=2 |

### latency p99 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 5.994 [5.987-6.000] n=2 | 12.112 [12.098-12.127] n=2 | 23.988 [23.977-23.999] n=2 |
| M* main | 11.840 [11.828-11.853] n=2 | 23.016 [23.003-23.029] n=2 | 45.815 [45.776-45.854] n=2 |
| vllm-omni 0.30.0 | 7.176 [7.172-7.180] n=2 | 14.097 [14.070-14.124] n=2 | 27.620 [27.618-27.623] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 5.958 [5.949-5.966] n=2 | 11.856 [11.841-11.871] n=2 | 23.167 [23.145-23.189] n=2 |
| M* main | 11.738 [11.730-11.746] n=2 | 22.858 [22.806-22.911] n=2 | 45.180 [45.142-45.218] n=2 |
| vllm-omni 0.30.0 | 7.119 [7.117-7.120] n=2 | 13.546 [13.533-13.559] n=2 | 27.202 [27.176-27.228] n=2 |

M* PR img/s relative to each arm:

- M* PR vs M* main: c=1 +96.7%, c=2 +91.3%, c=4 +94.5%
- M* PR vs vllm-omni 0.30.0: c=1 +19.6%, c=2 +14.6%, c=4 +17.2%

## edit: image edit, 1024x1024 input, 50 steps, cfg_text 4.0 / cfg_img 2.0

### img/s

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 0.075 [0.075-0.075] n=2 | 0.079 [0.078-0.079] n=2 |
| M* main | 0.073 [0.073-0.073] n=2 | 0.077 [0.077-0.078] n=2 |
| vllm-omni 0.30.0 | FAILED (n=2) | - |

### req/s

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 0.075 [0.075-0.075] n=2 | 0.079 [0.078-0.079] n=2 |
| M* main | 0.073 [0.073-0.073] n=2 | 0.077 [0.077-0.078] n=2 |
| vllm-omni 0.30.0 | FAILED (n=2) | - |

### latency p50 (s)

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 13.286 [13.258-13.314] n=2 | 50.396 [50.326-50.466] n=2 |
| M* main | 13.659 [13.655-13.663] n=2 | 51.140 [51.127-51.152] n=2 |
| vllm-omni 0.30.0 | FAILED (n=2) | - |

### latency p99 (s)

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 13.335 [13.312-13.358] n=2 | 52.172 [52.070-52.274] n=2 |
| M* main | 13.701 [13.696-13.705] n=2 | 53.025 [52.712-53.338] n=2 |
| vllm-omni 0.30.0 | FAILED (n=2) | - |

### TTFB p50 (s)

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 13.275 [13.248-13.303] n=2 | 50.384 [50.314-50.455] n=2 |
| M* main | 13.652 [13.648-13.657] n=2 | 51.133 [51.119-51.146] n=2 |
| vllm-omni 0.30.0 | FAILED (n=2) | - |

M* PR img/s relative to each arm:

- M* PR vs M* main: c=1 +2.9%, c=4 +1.4%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_pr | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 25.583267211914062, "psnr_vs_ref_median": null, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": null, "parity": true} |
| mstar_main | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 25.634765625, "psnr_vs_ref_median": 35.6055838698495, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 41.345490125615136, "parity": true} |
| mstar_pr_b | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 25.591659545898438, "psnr_vs_ref_median": 99.0, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 99.0, "parity": true} |

Profile: phases_t2i_c2.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

