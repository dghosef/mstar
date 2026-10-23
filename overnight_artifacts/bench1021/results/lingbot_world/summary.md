# lingbot_world: LingBot-World-V2 1.3B causal-fast, 480x832, 93 frames (examples/03)



Arms: M* PR = config; M* main = config; Robbyant/lingbot-world-v2 official WanI2VCausal @1895d300d8ac (offline batch) = default
Competitor choice: SGLang 0.5.22 and vllm-omni 0.30 support only the 14B LingBot-World-V2 checkpoint (vllm-omni's recipe: 1.3B 'not claimed'); diffusers has no LingBot pipeline. The official repo runs offline, as M*'s parity oracle runs it (M* venv + FA2), c=1 only (M* serves one rollout per GPU by design); it decodes the whole clip at the end, so only total fps compares (no first-chunk time)

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## rollout: one 93-frame rollout (13 + 5x16 frames streamed at 16 fps); frames/s

### frames/s

| arm | c=1 |
|---|---:|
| M* PR | 11.161 [11.158-11.163] n=2 |
| M* main | 9.361 [9.351-9.371] n=2 |
| Robbyant/lingbot-world-v2 official WanI2VCausal @1895d300d8ac (offline batch) | 9.119 [9.107-9.132] n=2 |

### req/s

| arm | c=1 |
|---|---:|
| M* PR | 0.120 [0.120-0.120] n=2 |
| M* main | 0.101 [0.101-0.101] n=2 |
| Robbyant/lingbot-world-v2 official WanI2VCausal @1895d300d8ac (offline batch) | 0.098 [0.098-0.098] n=2 |

### latency p50 (s)

| arm | c=1 |
|---|---:|
| M* PR | 8.298 [8.297-8.299] n=2 |
| M* main | 9.907 [9.898-9.916] n=2 |
| Robbyant/lingbot-world-v2 official WanI2VCausal @1895d300d8ac (offline batch) | 10.191 [10.177-10.205] n=2 |

### latency p99 (s)

| arm | c=1 |
|---|---:|
| M* PR | 8.384 [8.383-8.385] n=2 |
| M* main | 9.973 [9.952-9.994] n=2 |
| Robbyant/lingbot-world-v2 official WanI2VCausal @1895d300d8ac (offline batch) | 10.234 [10.221-10.246] n=2 |

### TTFB p50 (s)

| arm | c=1 |
|---|---:|
| M* PR | 1.305 [1.303-1.306] n=2 |
| M* main | 1.493 [1.492-1.495] n=2 |
| Robbyant/lingbot-world-v2 official WanI2VCausal @1895d300d8ac (offline batch) | - |

M* PR frames/s relative to each arm:

- M* PR vs M* main: c=1 +19.2%
- M* PR vs Robbyant/lingbot-world-v2 official WanI2VCausal @1895d300d8ac (offline batch): c=1 +22.4%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_pr | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": null, "parity": true} |
| lingbot_official__default | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 7.101262811607892, "parity": false} |
| mstar_main | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 70.428229973582, "parity": true} |
| mstar_pr_b | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 99.0, "parity": true} |

Profile: phases_rollout_c1.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

