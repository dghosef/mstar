# wan22: tag main

Arm configs: M* PR = config; M* main = config; vllm-omni = full; SGLang = base; diffusers (tuned, offline batch) = regional_dyn_fa3

## Throughput (videos/min), steps=20: mean [min-max] over rounds; n rounds; failed

| arm | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| M* PR | 2.667 [2.664-2.670] n=3 | 2.712 [2.710-2.713] n=3 | 2.724 [2.722-2.725] n=3 | 2.730 [2.729-2.730] n=3 |
| M* main | 2.433 [2.430-2.435] n=3 | 2.487 [2.486-2.489] n=3 | 2.501 [2.500-2.502] n=3 | 2.505 [2.504-2.506] n=3 |
| vllm-omni | 2.632 [2.632-2.632] n=3 | 2.685 [2.683-2.687] n=3 | 2.693 [2.692-2.693] n=3 | 2.697 [2.696-2.698] n=3 |
| SGLang | 2.424 [2.424-2.425] n=3 | 2.439 [2.438-2.441] n=3 | 2.440 [2.439-2.441] n=3 | 2.442 [2.441-2.444] n=3 |
| diffusers (tuned, offline batch) | 2.409 [2.407-2.411] n=3 | 2.402 [2.400-2.403] n=3 | 2.405 [2.404-2.406] n=3 | 2.399 [2.397-2.401] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| M* main | 1.10x | 1.09x | 1.09x | 1.09x |
| vllm-omni | 1.01x | 1.01x | 1.01x | 1.01x |
| SGLang | 1.10x | 1.11x | 1.12x | 1.12x |
| diffusers (tuned, offline batch) | 1.11x | 1.13x | 1.13x | 1.14x |

## B=1 latency (s), steps=20, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 22.434 [22.411-22.461] | 22.362 | 22.610 | 3 | 3 | 0 |
| M* main | 24.608 [24.587-24.631] | 24.519 | 24.813 | 3 | 3 | 0 |
| vllm-omni | 22.726 [22.720-22.730] | 22.727 | 22.770 | 3 | 3 | 0 |
| SGLang | 24.577 [24.563-24.601] | 24.599 | 24.701 | 3 | 3 | 0 |
| diffusers (tuned, offline batch) | 24.910 [24.888-24.931] | 24.902 | 24.960 | 3 | 3 | 0 |

## Throughput (videos/min), steps=50: mean [min-max] over rounds; n rounds; failed

| arm | c=1 |
|---|---:|
| M* PR | 1.156 [1.155-1.157] n=3 |
| M* main | 1.053 [1.052-1.054] n=3 |
| vllm-omni | 1.160 [1.159-1.161] n=3 |
| SGLang | 1.140 [1.138-1.141] n=3 |
| diffusers (tuned, offline batch) | 1.106 [1.105-1.106] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 |
|---|---:|
| M* main | 1.10x |
| vllm-omni | 1.00x |
| SGLang | 1.01x |
| diffusers (tuned, offline batch) | 1.05x |

## B=1 latency (s), steps=50, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 51.839 [51.791-51.879] | 51.785 | 51.994 | 3 | 3 | 0 |
| M* main | 56.937 [56.883-56.986] | 56.894 | 57.095 | 3 | 3 | 0 |
| vllm-omni | 51.645 [51.604-51.678] | 51.658 | 51.680 | 3 | 3 | 0 |
| SGLang | 52.473 [52.419-52.545] | 52.450 | 52.551 | 3 | 3 | 0 |
| diffusers (tuned, offline batch) | 54.267 [54.245-54.285] | 54.318 | 54.352 | 3 | 3 | 0 |

1-min load average at cell start: median 1.7, max 7.2 over 75 cells (192 vCPUs).
