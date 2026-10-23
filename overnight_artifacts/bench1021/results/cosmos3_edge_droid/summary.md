# cosmos3_edge_droid: Cosmos3-Edge DROID policy: 32-action chunks (4 steps, gs 3.0, shift 5.0)



Arms: M* PR = config; M* main = config; SGLang diffusion 0.5.22 (/v1/actions/generations) = b1
Competitor choice: SGLang 0.5.22 serves the DROID policy on /v1/actions/generations (every parameter sent: its defaults are 35 steps / gs 1.0 / horizon 16). It rewraps the task into a JSON caption (M* uses the raw text): not matchable, annotated. M* is driven over HTTP /generate here (the PR measured its WebSocket pipelining); the official cosmos-framework policy server takes a multi-camera RoboLab format and can't take the same request

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## policy: one observation + task -> one 32x10 action chunk (closed loop over HTTP)

### chunks/s

| arm | c=1 | c=4 | c=8 |
|---|---:|---:|---:|
| M* PR | 3.898 [3.896-3.901] n=2 | 4.646 [4.631-4.660] n=2 | 4.728 [4.706-4.750] n=2 |
| M* main | 3.878 [3.867-3.888] n=2 | 3.353 [3.353-3.353] n=2 | 3.242 [3.216-3.269] n=2 |
| SGLang diffusion 0.5.22 (/v1/actions/generations) | 3.122 [3.099-3.144] n=2 | 3.880 [3.873-3.887] n=2 | 3.894 [3.888-3.900] n=2 |

### req/s

| arm | c=1 | c=4 | c=8 |
|---|---:|---:|---:|
| M* PR | 3.898 [3.896-3.901] n=2 | 4.646 [4.631-4.660] n=2 | 4.728 [4.706-4.750] n=2 |
| M* main | 3.878 [3.867-3.888] n=2 | 3.353 [3.353-3.353] n=2 | 3.242 [3.216-3.269] n=2 |
| SGLang diffusion 0.5.22 (/v1/actions/generations) | 3.122 [3.099-3.144] n=2 | 3.880 [3.873-3.887] n=2 | 3.894 [3.888-3.900] n=2 |

### latency p50 (s)

| arm | c=1 | c=4 | c=8 |
|---|---:|---:|---:|
| M* PR | 0.256 [0.256-0.256] n=2 | 0.856 [0.853-0.860] n=2 | 1.688 [1.678-1.699] n=2 |
| M* main | 0.258 [0.256-0.260] n=2 | 1.184 [1.174-1.195] n=2 | 2.475 [2.454-2.497] n=2 |
| SGLang diffusion 0.5.22 (/v1/actions/generations) | 0.320 [0.317-0.322] n=2 | 1.021 [1.018-1.024] n=2 | 2.044 [2.039-2.049] n=2 |

### latency p99 (s)

| arm | c=1 | c=4 | c=8 |
|---|---:|---:|---:|
| M* PR | 0.257 [0.257-0.258] n=2 | 0.981 [0.970-0.992] n=2 | 1.889 [1.862-1.916] n=2 |
| M* main | 0.261 [0.261-0.262] n=2 | 1.351 [1.292-1.410] n=2 | 2.620 [2.601-2.640] n=2 |
| SGLang diffusion 0.5.22 (/v1/actions/generations) | 0.331 [0.327-0.336] n=2 | 1.098 [1.095-1.101] n=2 | 2.081 [2.070-2.092] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=4 | c=8 |
|---|---:|---:|---:|
| M* PR | 0.256 [0.256-0.256] n=2 | 0.856 [0.853-0.859] n=2 | 1.688 [1.678-1.698] n=2 |
| M* main | 0.258 [0.256-0.260] n=2 | 1.184 [1.173-1.195] n=2 | 2.475 [2.454-2.496] n=2 |
| SGLang diffusion 0.5.22 (/v1/actions/generations) | 0.320 [0.317-0.322] n=2 | 1.021 [1.018-1.024] n=2 | 2.044 [2.039-2.049] n=2 |

M* PR chunks/s relative to each arm:

- M* PR vs M* main: c=1 +0.5%, c=4 +38.5%, c=8 +45.8%
- M* PR vs SGLang diffusion 0.5.22 (/v1/actions/generations): c=1 +24.9%, c=4 +19.7%, c=8 +21.4%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_main | PASS | policy: {"n": 8, "missing": 0, "failed": 0, "bad": 0, "parity": null} |
| mstar_pr | PASS | policy: {"n": 8, "missing": 0, "failed": 0, "bad": 0, "parity": null} |
| mstar_pr_b | PASS | policy: {"n": 8, "missing": 0, "failed": 0, "bad": 0, "parity": null} |
| sglang__b1 | PASS | policy: {"n": 8, "missing": 0, "failed": 0, "bad": 0, "parity": null} |
| sglang__b8 | PASS | policy: {"n": 8, "missing": 0, "failed": 0, "bad": 0, "parity": null} |

Profile: phases_policy_c4.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

