# vjepa2: V-JEPA 2 ViT-L (fpc64-256): encoder + predictor latents, 64 frames



Arms: M* PR = config; M* main = config; HF transformers 5.20.2 VJEPA2Model (SDPA) (offline batch) = default
Competitor choice: no engine serves V-JEPA 2; HF transformers VJEPA2Model (full masks: the same [8192,1024] output as M*'s prefill_video) runs offline in batches, decode + preprocess + D2H included; M* returns its 32 MiB float32 output over HTTP (binary framing)

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## clip: 64-frame 720p clip -> predictor latents [8192,1024] f32 (32 MiB) per request

### req/s

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 1.408 [1.368-1.448] n=2 | 1.898 [1.831-1.965] n=2 | 1.900 [1.844-1.956] n=2 | 1.970 [1.968-1.972] n=2 |
| M* main | 0.696 [0.691-0.700] n=2 | 0.791 [0.783-0.798] n=2 | 0.787 [0.778-0.795] n=2 | 0.786 [0.780-0.792] n=2 |
| HF transformers 5.20.2 VJEPA2Model (SDPA) (offline batch) | 1.833 [1.799-1.866] n=2 | 1.895 [1.839-1.951] n=2 | 1.987 [1.980-1.994] n=2 | 2.016 [2.015-2.016] n=2 |

### req/s

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 1.408 [1.368-1.448] n=2 | 1.898 [1.831-1.965] n=2 | 1.900 [1.844-1.956] n=2 | 1.970 [1.968-1.972] n=2 |
| M* main | 0.696 [0.691-0.700] n=2 | 0.791 [0.783-0.798] n=2 | 0.787 [0.778-0.795] n=2 | 0.786 [0.780-0.792] n=2 |
| HF transformers 5.20.2 VJEPA2Model (SDPA) (offline batch) | 1.833 [1.799-1.866] n=2 | 1.895 [1.839-1.951] n=2 | 1.987 [1.980-1.994] n=2 | 2.016 [2.015-2.016] n=2 |

### latency p50 (s)

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 0.697 [0.662-0.732] n=2 | 2.069 [2.002-2.137] n=2 | 4.171 [4.026-4.315] n=2 | 8.041 [8.038-8.045] n=2 |
| M* main | 1.429 [1.418-1.439] n=2 | 5.021 [4.993-5.050] n=2 | 10.130 [10.036-10.223] n=2 | 20.291 [20.166-20.417] n=2 |
| HF transformers 5.20.2 VJEPA2Model (SDPA) (offline batch) | 0.551 [0.541-0.561] n=2 | 2.054 [2.041-2.067] n=2 | 4.015 [4.011-4.020] n=2 | 7.933 [7.932-7.934] n=2 |

### latency p99 (s)

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 0.764 [0.757-0.771] n=2 | 2.432 [2.387-2.477] n=2 | 4.572 [4.407-4.737] n=2 | 8.297 [8.215-8.378] n=2 |
| M* main | 1.478 [1.466-1.490] n=2 | 5.871 [5.768-5.973] n=2 | 10.733 [10.569-10.896] n=2 | 20.563 [20.340-20.785] n=2 |
| HF transformers 5.20.2 VJEPA2Model (SDPA) (offline batch) | 0.594 [0.593-0.595] n=2 | 2.251 [2.090-2.413] n=2 | 4.145 [4.062-4.228] n=2 | 7.999 [7.971-8.027] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 0.587 [0.554-0.620] n=2 | 1.958 [1.890-2.026] n=2 | 4.055 [3.913-4.197] n=2 | 7.932 [7.930-7.935] n=2 |
| M* main | 1.322 [1.316-1.327] n=2 | 4.917 [4.889-4.945] n=2 | 10.026 [9.930-10.122] n=2 | 20.188 [20.063-20.313] n=2 |
| HF transformers 5.20.2 VJEPA2Model (SDPA) (offline batch) | - | - | - | - |

M* PR req/s relative to each arm:

- M* PR vs M* main: c=1 +102.4%, c=4 +140.1%, c=8 +141.6%, c=16 +150.6%
- M* PR vs HF transformers 5.20.2 VJEPA2Model (SDPA) (offline batch): c=1 -23.2%, c=4 +0.2%, c=8 -4.4%, c=16 -2.3%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| hf_transformers__default | PASS | clip: {"n": 4, "missing": 0, "failed": 0, "nonfinite": 0, "repeated_chunks": 0, "cos_mean": 0.9947251604467816, "cos_min": 0.9947251604467816, "parity": true} |
| mstar_main | PASS | clip: {"n": 4, "missing": 0, "failed": 0, "nonfinite": 0, "repeated_chunks": 0, "cos_mean": 1.0, "cos_min": 1.0, "parity": true} |
| mstar_pr | PASS | clip: {"n": 4, "missing": 0, "failed": 0, "nonfinite": 0, "repeated_chunks": 0, "cos_mean": 1.0, "cos_min": 1.0, "parity": true} |
| mstar_pr_b | PASS | clip: {"n": 4, "missing": 0, "failed": 0, "nonfinite": 0, "repeated_chunks": 0, "cos_mean": 1.0, "cos_min": 1.0, "parity": true} |

Profile: phases_clip_c1.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

