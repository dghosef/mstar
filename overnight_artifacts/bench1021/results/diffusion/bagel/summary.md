# bagel: BAGEL-7B-MoT on 1 GPU



Arms: M* PR = config; M* main = config; vllm-omni 0.30.0 = single_step4; vLLM 0.30.0 (understanding only) = default
Competitor choice: vllm-omni 0.30.0 serves BAGEL t2i/edit (since v0.12.0rc1) through /v1/chat/completions with every CFG knob sent explicitly (its /v1/images routes drop them; its defaults differ: cfg_img 1.5, shift 1.0 on one path); vLLM 0.30.0 core serves understanding (streams, so it is the stronger understanding arm). SGLang/sglang-omni/diffusers have no BAGEL. The official repo is serial and pinned to py3.10/torch 2.5 (not queued). M*'s cfg_prune runs the same CFG branches as the reference (41 dB vs base, inside the 40 dB same-code floor)

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## t2i: text -> image 1024x1024, 50 steps, cfg_text 4.0 (BAGEL defaults), PartiPrompts

### img/s

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 0.102 [0.101-0.102] n=2 | 0.102 [0.102-0.102] n=2 | 0.102 [0.102-0.102] n=2 |
| M* main | 0.061 [0.061-0.062] n=2 | 0.062 [0.062-0.063] n=2 | 0.063 [0.062-0.063] n=2 |
| vllm-omni 0.30.0 | 0.080 [0.080-0.080] n=2 | 0.082 [0.082-0.082] n=2 | 0.082 [0.082-0.083] n=2 |
| vLLM 0.30.0 (understanding only) | - | - | - |

### req/s

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 0.102 [0.101-0.102] n=2 | 0.102 [0.102-0.102] n=2 | 0.102 [0.102-0.102] n=2 |
| M* main | 0.061 [0.061-0.062] n=2 | 0.062 [0.062-0.063] n=2 | 0.063 [0.062-0.063] n=2 |
| vllm-omni 0.30.0 | 0.080 [0.080-0.080] n=2 | 0.082 [0.082-0.082] n=2 | 0.082 [0.082-0.083] n=2 |
| vLLM 0.30.0 (understanding only) | - | - | - |

### latency p50 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 9.818 [9.806-9.831] n=2 | 19.497 [19.449-19.545] n=2 | 39.059 [38.963-39.155] n=2 |
| M* main | 16.220 [16.184-16.256] n=2 | 31.937 [31.894-31.979] n=2 | 63.979 [63.914-64.043] n=2 |
| vllm-omni 0.30.0 | 12.491 [12.458-12.524] n=2 | 24.301 [24.224-24.377] n=2 | 48.138 [48.027-48.249] n=2 |
| vLLM 0.30.0 (understanding only) | - | - | - |

### latency p99 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 9.840 [9.835-9.845] n=2 | 19.634 [19.591-19.678] n=2 | 39.203 [39.117-39.288] n=2 |
| M* main | 16.302 [16.264-16.340] n=2 | 32.443 [32.314-32.572] n=2 | 64.727 [64.568-64.886] n=2 |
| vllm-omni 0.30.0 | 12.544 [12.537-12.551] n=2 | 24.459 [24.360-24.558] n=2 | 49.945 [48.856-51.034] n=2 |
| vLLM 0.30.0 (understanding only) | - | - | - |

### TTFB p50 (s)

| arm | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| M* PR | 9.809 [9.796-9.822] n=2 | 19.486 [19.438-19.533] n=2 | 39.048 [38.953-39.143] n=2 |
| M* main | 16.215 [16.180-16.250] n=2 | 31.930 [31.887-31.973] n=2 | 63.972 [63.908-64.036] n=2 |
| vllm-omni 0.30.0 | 12.483 [12.452-12.515] n=2 | 24.294 [24.217-24.372] n=2 | 48.125 [48.013-48.236] n=2 |
| vLLM 0.30.0 (understanding only) | - | - | - |

M* PR img/s relative to each arm:

- M* PR vs M* main: c=1 +65.1%, c=2 +63.7%, c=4 +63.6%
- M* PR vs vllm-omni 0.30.0: c=1 +27.0%, c=2 +24.4%, c=4 +24.1%

## edit: image edit, 1024x1024 input, 50 steps, cfg_text 4.0 / cfg_img 2.0

### img/s

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 0.052 [0.052-0.052] n=2 | 0.053 [0.053-0.053] n=2 |
| M* main | 0.051 [0.051-0.051] n=2 | 0.052 [0.052-0.052] n=2 |
| vllm-omni 0.30.0 | 0.044 [0.044-0.044] n=2 | 0.046 [0.046-0.046] n=2 |
| vLLM 0.30.0 (understanding only) | - | - |

### req/s

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 0.052 [0.052-0.052] n=2 | 0.053 [0.053-0.053] n=2 |
| M* main | 0.051 [0.051-0.051] n=2 | 0.052 [0.052-0.052] n=2 |
| vllm-omni 0.30.0 | 0.044 [0.044-0.044] n=2 | 0.046 [0.046-0.046] n=2 |
| vLLM 0.30.0 (understanding only) | - | - |

### latency p50 (s)

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 19.063 [19.031-19.094] n=2 | 74.979 [74.679-75.279] n=2 |
| M* main | 19.430 [19.411-19.448] n=2 | 74.826 [74.537-75.114] n=2 |
| vllm-omni 0.30.0 | 22.820 [22.776-22.865] n=2 | 85.997 [85.813-86.180] n=2 |
| vLLM 0.30.0 (understanding only) | - | - |

### latency p99 (s)

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 19.076 [19.047-19.105] n=2 | 76.911 [76.738-77.084] n=2 |
| M* main | 19.480 [19.457-19.503] n=2 | 78.279 [78.181-78.377] n=2 |
| vllm-omni 0.30.0 | 22.976 [22.958-22.995] n=2 | 87.106 [87.047-87.165] n=2 |
| vLLM 0.30.0 (understanding only) | - | - |

### TTFB p50 (s)

| arm | c=1 | c=4 |
|---|---:|---:|
| M* PR | 19.052 [19.021-19.083] n=2 | 74.968 [74.669-75.267] n=2 |
| M* main | 19.422 [19.404-19.441] n=2 | 74.820 [74.531-75.108] n=2 |
| vllm-omni 0.30.0 | 22.813 [22.769-22.858] n=2 | 85.986 [85.804-86.168] n=2 |
| vLLM 0.30.0 (understanding only) | - | - |

M* PR img/s relative to each arm:

- M* PR vs M* main: c=1 +2.0%, c=4 +0.4%
- M* PR vs vllm-omni 0.30.0: c=1 +19.8%, c=4 +13.7%

## und: image understanding (food101), 64 tokens ignore_eos, streamed

### img/s

| arm | c=1 | c=4 | c=16 |
|---|---:|---:|---:|
| M* PR | 2.022 [2.021-2.022] n=2 | 5.607 [5.605-5.610] n=2 | 19.925 [19.921-19.929] n=2 |
| M* main | 2.451 [2.449-2.453] n=2 | 5.239 [5.239-5.239] n=2 | 16.242 [16.149-16.334] n=2 |
| vllm-omni 0.30.0 | - | - | - |
| vLLM 0.30.0 (understanding only) | 1.597 [1.596-1.599] n=2 | 2.721 [2.719-2.723] n=2 | 3.291 [3.291-3.292] n=2 |

### req/s

| arm | c=1 | c=4 | c=16 |
|---|---:|---:|---:|
| M* PR | 2.022 [2.021-2.022] n=2 | 5.607 [5.605-5.610] n=2 | 19.925 [19.921-19.929] n=2 |
| M* main | 2.451 [2.449-2.453] n=2 | 5.239 [5.239-5.239] n=2 | 16.242 [16.149-16.334] n=2 |
| vllm-omni 0.30.0 | - | - | - |
| vLLM 0.30.0 (understanding only) | 1.597 [1.596-1.599] n=2 | 2.721 [2.719-2.723] n=2 | 3.291 [3.291-3.292] n=2 |

### latency p50 (s)

| arm | c=1 | c=4 | c=16 |
|---|---:|---:|---:|
| M* PR | 0.494 [0.494-0.494] n=2 | 0.716 [0.712-0.720] n=2 | 0.797 [0.796-0.797] n=2 |
| M* main | 0.407 [0.407-0.407] n=2 | 0.748 [0.729-0.766] n=2 | 0.976 [0.972-0.980] n=2 |
| vllm-omni 0.30.0 | - | - | - |
| vLLM 0.30.0 (understanding only) | 0.630 [0.630-0.630] n=2 | 1.479 [1.476-1.483] n=2 | 4.817 [4.811-4.824] n=2 |

### latency p99 (s)

| arm | c=1 | c=4 | c=16 |
|---|---:|---:|---:|
| M* PR | 0.563 [0.563-0.563] n=2 | 0.985 [0.979-0.991] n=2 | 0.858 [0.838-0.879] n=2 |
| M* main | 0.475 [0.473-0.477] n=2 | 1.046 [1.045-1.046] n=2 | 1.158 [1.101-1.215] n=2 |
| vllm-omni 0.30.0 | - | - | - |
| vLLM 0.30.0 (understanding only) | 0.645 [0.644-0.646] n=2 | 1.691 [1.687-1.694] n=2 | 8.692 [8.688-8.697] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=4 | c=16 |
|---|---:|---:|---:|
| M* PR | 0.107 [0.107-0.107] n=2 | 0.180 [0.177-0.182] n=2 | 0.391 [0.389-0.392] n=2 |
| M* main | 0.118 [0.117-0.118] n=2 | 0.212 [0.208-0.216] n=2 | 0.535 [0.535-0.536] n=2 |
| vllm-omni 0.30.0 | - | - | - |
| vLLM 0.30.0 (understanding only) | 0.299 [0.299-0.299] n=2 | 0.719 [0.716-0.721] n=2 | 0.502 [0.501-0.503] n=2 |

M* PR img/s relative to each arm:

- M* PR vs M* main: c=1 -17.5%, c=4 +7.0%, c=16 +22.7%
- M* PR vs vLLM 0.30.0 (understanding only): c=1 +26.6%, c=4 +106.1%, c=16 +505.4%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_pr | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 25.595474243164062, "psnr_vs_ref_median": null, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": null, "parity": true}; und: {"n": 32, "missing": 0, "failed": 0, "empty": 0, "overlap_vs_ref": 1.0, "parity": true} |
| mstar_main | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 25.621795654296875, "psnr_vs_ref_median": 39.48687188457049, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 41.59601062373416, "parity": true}; und: {"n": 32, "missing": 0, "failed": 0, "empty": 0, "overlap_vs_ref": 0.6221826631052936, "parity": true} |
| mstar_pr_b | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 25.652694702148438, "psnr_vs_ref_median": 99.0, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 46.97353073315347, "parity": true}; und: {"n": 32, "missing": 0, "failed": 0, "empty": 0, "overlap_vs_ref": 0.7560092703365919, "parity": true} |
| vllmomni__single | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 26.241302490234375, "psnr_vs_ref_median": 9.521640953195707, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 11.298525238588974, "parity": false} |
| vllmomni__single_step4 | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 26.593017578125, "psnr_vs_ref_median": 9.519278799409644, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 11.297733332749498, "parity": false} |
| vllmomni__two_stage | PASS | t2i: {"n": 16, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": 25.230789184570312, "psnr_vs_ref_median": 9.983850091144982, "parity": true}; edit: {"n": 8, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 13.965901028535221, "parity": false} |
| vllm__default | PASS | und: {"n": 32, "missing": 0, "failed": 0, "empty": 0, "overlap_vs_ref": 0.26095707956890907, "parity": false} |

Profile: phases_edit_c4.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

