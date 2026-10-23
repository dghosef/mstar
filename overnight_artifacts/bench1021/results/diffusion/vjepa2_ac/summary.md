# vjepa2_ac: V-JEPA 2-AC ViT-g: 4-step action-conditioned rollout



Arms: M* PR = config; M* main = config; facebookresearch/vjepa2 official torch.hub vjepa2_ac_vit_giant @204698b (offline batch) = default
Competitor choice: no engine serves V-JEPA 2 (no jepa model in vLLM/SGLang); the official repo's AC predictor rollout (growing history, no KV cache: as shipped) runs offline in batches with M*-matched inputs and the request's actions/states; mstar-main's non-streaming rollout returns the last latent H times (fixed in the PR) and is flagged by the gate's repeated-chunk check

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## clip: 8-frame clip -> 4 rollout latents [256,1408] f32 (actions/states ramps, H=4)

### req/s

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 1.821 [1.773-1.870] n=2 | 2.026 [2.001-2.050] n=2 | 2.049 [2.028-2.070] n=2 | 2.066 [2.042-2.089] n=2 |
| M* main | 0.756 [0.748-0.763] n=2 | 0.819 [0.814-0.824] n=2 | 0.819 [0.810-0.828] n=2 | 0.815 [0.807-0.823] n=2 |
| facebookresearch/vjepa2 official torch.hub vjepa2_ac_vit_giant @204698b (offline batch) | 2.177 [2.163-2.191] n=2 | 6.030 [6.018-6.042] n=2 | 7.991 [7.907-8.075] n=2 | 9.524 [9.418-9.631] n=2 |

### req/s

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 1.821 [1.773-1.870] n=2 | 2.026 [2.001-2.050] n=2 | 2.049 [2.028-2.070] n=2 | 2.066 [2.042-2.089] n=2 |
| M* main | 0.756 [0.748-0.763] n=2 | 0.819 [0.814-0.824] n=2 | 0.819 [0.810-0.828] n=2 | 0.815 [0.807-0.823] n=2 |
| facebookresearch/vjepa2 official torch.hub vjepa2_ac_vit_giant @204698b (offline batch) | 2.177 [2.163-2.191] n=2 | 6.030 [6.018-6.042] n=2 | 7.991 [7.907-8.075] n=2 | 9.524 [9.418-9.631] n=2 |

### latency p50 (s)

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 0.535 [0.519-0.550] n=2 | 1.961 [1.937-1.986] n=2 | 3.906 [3.854-3.959] n=2 | 7.733 [7.649-7.818] n=2 |
| M* main | 1.318 [1.308-1.328] n=2 | 4.878 [4.855-4.901] n=2 | 9.753 [9.640-9.867] n=2 | 19.603 [19.415-19.791] n=2 |
| facebookresearch/vjepa2 official torch.hub vjepa2_ac_vit_giant @204698b (offline batch) | 0.458 [0.458-0.459] n=2 | 0.660 [0.659-0.660] n=2 | 1.000 [0.988-1.011] n=2 | 1.684 [1.664-1.703] n=2 |

### latency p99 (s)

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 0.645 [0.640-0.650] n=2 | 2.304 [2.091-2.518] n=2 | 4.128 [4.030-4.227] n=2 | 8.014 [7.979-8.050] n=2 |
| M* main | 1.379 [1.360-1.398] n=2 | 5.006 [4.967-5.044] n=2 | 10.023 [9.968-10.077] n=2 | 20.003 [19.758-20.249] n=2 |
| facebookresearch/vjepa2 official torch.hub vjepa2_ac_vit_giant @204698b (offline batch) | 0.488 [0.487-0.489] n=2 | 0.709 [0.693-0.724] n=2 | 1.031 [1.022-1.040] n=2 | 1.724 [1.689-1.759] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|
| M* PR | 0.521 [0.505-0.536] n=2 | 1.943 [1.919-1.968] n=2 | 3.886 [3.838-3.935] n=2 | 7.716 [7.631-7.802] n=2 |
| M* main | 1.304 [1.293-1.315] n=2 | 4.865 [4.842-4.888] n=2 | 9.740 [9.627-9.853] n=2 | 19.589 [19.401-19.777] n=2 |
| facebookresearch/vjepa2 official torch.hub vjepa2_ac_vit_giant @204698b (offline batch) | - | - | - | - |

M* PR req/s relative to each arm:

- M* PR vs M* main: c=1 +141.0%, c=4 +147.4%, c=8 +150.3%, c=16 +153.4%
- M* PR vs facebookresearch/vjepa2 official torch.hub vjepa2_ac_vit_giant @204698b (offline batch): c=1 -16.3%, c=4 -66.4%, c=8 -74.4%, c=16 -78.3%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_main | FAIL (clip: FAIL (broken output)) | clip: {"n": 4, "missing": 0, "failed": 0, "nonfinite": 0, "repeated_chunks": 4, "cos_mean": 0.97995609151534, "cos_min": 0.97995609151534, "parity": false} |
| mstar_pr | PASS | clip: {"n": 4, "missing": 0, "failed": 0, "nonfinite": 0, "repeated_chunks": 0, "cos_mean": 0.9999999999999999, "cos_min": 0.9999999999999999, "parity": true} |
| mstar_pr_b | PASS | clip: {"n": 4, "missing": 0, "failed": 0, "nonfinite": 0, "repeated_chunks": 0, "cos_mean": 0.9999999999999999, "cos_min": 0.9999999999999999, "parity": true} |

Profile: phases_clip_c1.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

Profile: phases_clip_c16.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

