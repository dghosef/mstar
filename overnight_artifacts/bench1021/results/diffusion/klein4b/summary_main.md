# klein4b: tag main

Arm configs: M* PR = config; M* main = config; vllm-omni = full; SGLang = base; diffusers (tuned, offline batch) = regional_dyn_fa3_vae

## Throughput (img/s), steps=4: mean [min-max] over rounds; n rounds; failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* PR | 2.719 [2.706-2.736] n=3 | 3.294 [3.285-3.311] n=3 | 3.388 [3.384-3.394] n=3 | 3.385 [3.365-3.394] n=3 | 3.416 [3.398-3.426] n=3 |
| M* main | 2.644 [2.639-2.648] n=3 | 3.116 [3.056-3.189] n=3 | 3.306 [3.288-3.320] n=3 | 3.330 [3.318-3.351] n=3 | 3.361 [3.357-3.369] n=3 |
| vllm-omni | 2.094 [2.089-2.100] n=3 | 2.629 [2.615-2.641] n=3 | 2.668 [2.667-2.668] n=3 | 2.674 [2.672-2.678] n=3 | 2.672 [2.667-2.678] n=3 |
| SGLang | 3.522 [3.520-3.528] n=3 | 3.636 [3.635-3.638] n=3 | 3.635 [3.633-3.637] n=3 | 3.629 [3.626-3.635] n=3 | 3.624 [3.623-3.625] n=3 |
| diffusers (tuned, offline batch) | 3.226 [3.209-3.236] n=3 | 3.301 [3.299-3.303] n=3 | 3.332 [3.329-3.335] n=3 | 3.324 [3.320-3.328] n=3 | 3.326 [3.325-3.326] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* main | 1.03x | 1.06x | 1.02x | 1.02x | 1.02x |
| vllm-omni | 1.30x | 1.25x | 1.27x | 1.27x | 1.28x |
| SGLang | 0.77x | 0.91x | 0.93x | 0.93x | 0.94x |
| diffusers (tuned, offline batch) | 0.84x | 1.00x | 1.02x | 1.02x | 1.03x |

## B=1 latency (s), steps=4, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 0.338 [0.337-0.340] | 0.338 | 0.352 | 32 | 3 | 0 |
| M* main | 0.349 [0.349-0.350] | 0.349 | 0.361 | 32 | 3 | 0 |
| vllm-omni | 0.454 [0.452-0.455] | 0.454 | 0.466 | 32 | 3 | 0 |
| SGLang | 0.273 [0.272-0.273] | 0.273 | 0.277 | 32 | 3 | 0 |
| diffusers (tuned, offline batch) | 0.310 [0.309-0.312] | 0.310 | 0.317 | 32 | 3 | 0 |

1-min load average at cell start: median 4.0, max 16.6 over 75 cells (192 vCPUs).
