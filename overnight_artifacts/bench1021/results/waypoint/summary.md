# waypoint: Waypoint-1.5-1B 720p: 64-step interactive rollouts (256 frames each)



Arms: M* PR = config; M* main = config; Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) = default
Competitor choice: no serving engine has Waypoint; Overworld's only server (Biome) is single-session with JPEG frames, so the official world_engine library runs as an offline loop (bf16, its own compile + CUDA graphs, decode + .cpu() per chunk), b processes on one GPU for c=b (it has no batching); M* streams raw frames over HTTP. Different noise RNG: no pixel parity (sanity: non-flat frames)

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## rollout: seed image + 64 actions -> 64 chunks of 4 1280x720 frames (60 fps media)

### frames/s

| arm | c=1 | c=2 |
|---|---:|---:|
| M* PR | 88.399 [86.153-90.645] n=2 | 118.577 [117.643-119.511] n=2 |
| M* main | 90.294 [89.979-90.610] n=2 | 116.640 [116.547-116.734] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 126.872 [122.293-131.451] n=2 | 127.035 [125.185-128.884] n=2 |

### req/s

| arm | c=1 | c=2 |
|---|---:|---:|
| M* PR | 0.345 [0.337-0.354] n=2 | 0.463 [0.460-0.467] n=2 |
| M* main | 0.353 [0.351-0.354] n=2 | 0.456 [0.455-0.456] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 0.496 [0.478-0.513] n=2 | 0.496 [0.489-0.503] n=2 |

### latency p50 (s)

| arm | c=1 | c=2 |
|---|---:|---:|
| M* PR | 2.771 [2.731-2.811] n=2 | 3.809 [3.784-3.835] n=2 |
| M* main | 2.727 [2.725-2.730] n=2 | 3.840 [3.791-3.889] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 2.039 [1.929-2.148] n=2 | 3.946 [3.835-4.058] n=2 |

### latency p99 (s)

| arm | c=1 | c=2 |
|---|---:|---:|
| M* PR | 2.879 [2.852-2.906] n=2 | 5.481 [5.360-5.602] n=2 |
| M* main | 2.907 [2.845-2.968] n=2 | 5.627 [5.474-5.779] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 2.044 [1.932-2.155] n=2 | 4.168 [4.142-4.194] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=2 |
|---|---:|---:|
| M* PR | 0.083 [0.082-0.083] n=2 | 0.102 [0.094-0.109] n=2 |
| M* main | 0.086 [0.085-0.087] n=2 | 0.102 [0.095-0.109] n=2 |
| Overworldai world_engine 1.5.6 (official; one world per process) (offline batch) | 0.025 [0.023-0.026] n=2 | 0.049 [0.048-0.049] n=2 |

M* PR frames/s relative to each arm:

- M* PR vs M* main: c=1 -2.1%, c=2 +1.7%
- M* PR vs Overworldai world_engine 1.5.6 (official; one world per process) (offline batch): c=1 -30.3%, c=2 -6.7%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_pr | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": null, "parity": true} |
| mstar_main | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 21.92082155882547, "parity": false} |
| mstar_pr_b | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 99.0, "parity": true} |
| world_engine__default | PASS | rollout: {"n": 2, "missing": 0, "failed": 0, "undecodable": 0, "flat": 0, "static_video": 0, "clip_mean": null, "psnr_vs_ref_median": 20.927881475229377, "parity": false} |

Profile: phases_rollout_c1.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

