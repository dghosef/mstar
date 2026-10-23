# zimage: tag main

Arm configs: M* PR = config; M* main = config; vllm-omni = base; SGLang = bcg; diffusers (tuned, offline batch) = regional_dyn

## Throughput (img/s), steps=8: mean [min-max] over rounds; n rounds; failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* PR | 1.142 [1.140-1.144] n=3 | 1.192 [1.192-1.194] n=3 | 1.208 [1.206-1.209] n=3 | 1.214 [1.213-1.215] n=3 | 1.206 [1.205-1.208] n=3 |
| M* main | 1.123 [1.121-1.126] n=3 | 1.173 [1.170-1.176] n=3 | 1.104 [1.035-1.195] n=3 | 1.072 [1.040-1.105] n=3 | 1.174 [1.173-1.174] n=3 |
| vllm-omni | 1.035 [1.034-1.036] n=3 | 1.145 [1.144-1.146] n=3 | 1.149 [1.148-1.150] n=3 | 1.150 [1.150-1.151] n=3 | 1.149 [1.148-1.150] n=3 |
| SGLang | 1.260 [1.260-1.260] n=3 | 1.271 [1.271-1.271] n=3 | 1.274 [1.274-1.274] n=3 | 1.273 [1.273-1.274] n=3 | 1.271 [1.270-1.271] n=3 |
| diffusers (tuned, offline batch) | 1.120 [1.120-1.120] n=3 | 1.024 [1.023-1.026] n=3 | 1.098 [1.097-1.099] n=3 | 1.101 [1.100-1.102] n=3 | 1.081 [1.081-1.082] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* main | 1.02x | 1.02x | 1.09x | 1.13x | 1.03x |
| vllm-omni | 1.10x | 1.04x | 1.05x | 1.06x | 1.05x |
| SGLang | 0.91x | 0.94x | 0.95x | 0.95x | 0.95x |
| diffusers (tuned, offline batch) | 1.02x | 1.16x | 1.10x | 1.10x | 1.12x |

## B=1 latency (s), steps=8, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 0.846 [0.844-0.847] | 0.843 | 0.884 | 32 | 3 | 0 |
| M* main | 0.860 [0.858-0.862] | 0.858 | 0.894 | 32 | 3 | 0 |
| vllm-omni | 0.941 [0.940-0.943] | 0.940 | 0.972 | 32 | 3 | 0 |
| SGLang | 0.782 [0.782-0.783] | 0.781 | 0.805 | 32 | 3 | 0 |
| diffusers (tuned, offline batch) | 0.893 [0.892-0.893] | 0.887 | 0.997 | 32 | 3 | 0 |

1-min load average at cell start: median 1.9, max 19.4 over 75 cells (192 vCPUs).
