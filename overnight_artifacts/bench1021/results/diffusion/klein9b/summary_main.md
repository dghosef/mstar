# klein9b: tag main

Arm configs: M* PR = config; M* main = config; vllm-omni = full; SGLang = compile_b8; diffusers (tuned, offline batch) = regional_dyn_fa3_vae

## Throughput (img/s), steps=4: mean [min-max] over rounds; n rounds; failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* PR | 1.560 [1.557-1.564] n=3 | 1.719 [1.714-1.727] n=3 | 1.740 [1.736-1.744] n=3 | 1.748 [1.746-1.749] n=3 | 1.750 [1.749-1.751] n=3 |
| M* main | 1.537 [1.531-1.542] n=3 | 1.694 [1.693-1.694] n=3 | 1.721 [1.720-1.722] n=3 | 1.729 [1.729-1.731] n=3 | 1.733 [1.732-1.733] n=3 |
| vllm-omni | 1.354 [1.351-1.357] n=3 | 1.564 [1.563-1.565] n=3 | 1.568 [1.560-1.576] n=3 | 1.571 [1.570-1.572] n=3 | 1.569 [1.562-1.573] n=3 |
| SGLang | 1.719 [1.717-1.720] n=3 | 1.865 [1.863-1.865] n=3 | 1.816 [1.815-1.816] n=3 | 1.845 [1.843-1.846] n=3 | 1.856 [1.855-1.858] n=3 |
| diffusers (tuned, offline batch) | 1.762 [1.759-1.767] n=3 | 1.765 [1.764-1.766] n=3 | 1.775 [1.775-1.776] n=3 | 1.781 [1.779-1.782] n=3 | 1.777 [1.775-1.779] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* main | 1.02x | 1.01x | 1.01x | 1.01x | 1.01x |
| vllm-omni | 1.15x | 1.10x | 1.11x | 1.11x | 1.12x |
| SGLang | 0.91x | 0.92x | 0.96x | 0.95x | 0.94x |
| diffusers (tuned, offline batch) | 0.89x | 0.97x | 0.98x | 0.98x | 0.99x |

## B=1 latency (s), steps=4, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 0.611 [0.610-0.612] | 0.612 | 0.627 | 32 | 3 | 0 |
| M* main | 0.620 [0.619-0.621] | 0.620 | 0.634 | 32 | 3 | 0 |
| vllm-omni | 0.715 [0.714-0.716] | 0.715 | 0.727 | 32 | 3 | 0 |
| SGLang | 0.570 [0.570-0.571] | 0.571 | 0.578 | 32 | 3 | 0 |
| diffusers (tuned, offline batch) | 0.567 [0.566-0.569] | 0.567 | 0.579 | 32 | 3 | 0 |

1-min load average at cell start: median 2.4, max 5.8 over 75 cells (192 vCPUs).
