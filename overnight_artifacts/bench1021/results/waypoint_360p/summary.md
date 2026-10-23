# waypoint_360p: Waypoint-1.5-1B 360p: 64-step interactive rollouts (256 frames each)



Arms: M* PR = config; M* main = config; Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) = default
Competitor choice: no serving engine has Waypoint; Overworld's only server (Biome) is single-session with JPEG frames, so the official world_engine library runs as an offline loop (bf16, its own compile + CUDA graphs, decode + .cpu() per chunk), b processes on one GPU for c=b (it has no batching); M* streams raw frames over HTTP. Different noise RNG: no pixel parity (sanity: non-flat frames)

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## rollout: seed image + 64 actions -> 64 chunks of 4 640x360 frames (60 fps media)

### frames/s

| arm | c=1 | c=8 |
|---|---:|---:|
| M* PR | 233.320 [231.001-235.639] n=2 | 461.751 [457.877-465.624] n=2 |
| M* main | 234.607 [231.817-237.397] n=2 | 474.003 [471.435-476.570] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 253.332 [253.233-253.432] n=2 | 261.401 [260.621-262.182] n=2 |

### req/s

| arm | c=1 | c=8 |
|---|---:|---:|
| M* PR | 0.911 [0.902-0.920] n=2 | 1.804 [1.789-1.819] n=2 |
| M* main | 0.916 [0.906-0.927] n=2 | 1.852 [1.842-1.862] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 0.990 [0.989-0.990] n=2 | 1.021 [1.018-1.024] n=2 |

### latency p50 (s)

| arm | c=1 | c=8 |
|---|---:|---:|
| M* PR | 1.070 [1.065-1.076] n=2 | 4.146 [4.083-4.210] n=2 |
| M* main | 1.064 [1.057-1.072] n=2 | 3.916 [3.836-3.995] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 0.994 [0.994-0.994] n=2 | 7.750 [7.727-7.773] n=2 |

### latency p99 (s)

| arm | c=1 | c=8 |
|---|---:|---:|
| M* PR | 1.102 [1.102-1.102] n=2 | 5.521 [5.309-5.734] n=2 |
| M* main | 1.092 [1.091-1.093] n=2 | 5.421 [5.286-5.555] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 0.996 [0.994-0.997] n=2 | 7.761 [7.742-7.781] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=8 |
|---|---:|---:|
| M* PR | 0.042 [0.042-0.042] n=2 | 0.177 [0.165-0.189] n=2 |
| M* main | 0.042 [0.041-0.042] n=2 | 0.184 [0.154-0.213] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 0.014 [0.014-0.014] n=2 | 0.115 [0.115-0.115] n=2 |

M* PR frames/s relative to each arm:

- M* PR vs M* main: c=1 -0.5%, c=8 -2.6%
- M* PR vs Overworldai world_engine 1.5.6 (official; one world per process) (offline batch): c=1 -7.9%, c=8 +76.6%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_pr | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": null, "parity": true} |
| mstar_main | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 38.70191686874196, "parity": true} |
| mstar_pr_b | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 99.0, "parity": true} |
| world_engine__default | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 20.69701249551985, "parity": false} |

Profile: phases_rollout_c1.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

