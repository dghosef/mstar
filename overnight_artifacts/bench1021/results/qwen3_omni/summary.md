### qwen3_omni (Qwen/Qwen3-Omni-30B-A3B-Instruct, GPUs 0,1)

Output tok/s: mean (min-max) over rounds, each round a fresh server process; n = counted rounds. TTFT/TPOT p50 and p99 in ms, pooled over every request of the counted rounds. Configs: mstar-pr=['default'], mstar-main=['default'], vllm-omni=['default'], sglang-omni=['default']

#### omni_a2t

| c | mstar-pr tok/s | mstar-main tok/s | vllm-omni tok/s | sglang-omni tok/s | mstar-pr vs mstar-main | mstar-pr vs vllm-omni | mstar-pr vs sglang-omni | TTFT p50/p99 mstar-pr | TTFT p50/p99 mstar-main | TTFT p50/p99 vllm-omni | TTFT p50/p99 sglang-omni | TPOT p50/p99 mstar-pr | TPOT p50/p99 mstar-main | TPOT p50/p99 vllm-omni | TPOT p50/p99 sglang-omni |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | - | 45 (45-45) n=1 | - | 181 (181-181) n=1 | - | - | - | - | 92.6/2274.3 | - | 35.4/45.7 | - | 4.86/5.51 | - | 4.20/4.48 |
| 4 | - | 172 (172-172) n=1 | - | 391 (391-391) n=1 | - | - | - | - | 164.3/905.8 | - | 46.5/92.1 | - | 12.96/48.86 | - | 6.82/9.63 |
| 16 | - | 320 (320-320) n=1 | - | 900 (900-900) n=1 | - | - | - | - | 263.6/1730.5 | - | 69.6/130.8 | - | 32.31/148.70 | - | 14.90/21.37 |
| 32 | - | 560 (560-560) n=1 | - | 1,264 (1,264-1,264) n=1 | - | - | - | - | 356.5/885.1 | - | 71.8/286.8 | - | 37.76/112.45 | - | 20.69/28.22 |

#### omni_t2t

| c | mstar-pr tok/s | mstar-main tok/s | vllm-omni tok/s | sglang-omni tok/s | mstar-pr vs mstar-main | mstar-pr vs vllm-omni | mstar-pr vs sglang-omni | TTFT p50/p99 mstar-pr | TTFT p50/p99 mstar-main | TTFT p50/p99 vllm-omni | TTFT p50/p99 sglang-omni | TPOT p50/p99 mstar-pr | TPOT p50/p99 mstar-main | TPOT p50/p99 vllm-omni | TPOT p50/p99 sglang-omni |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 245 (245-245) n=3 | 216 (216-216) n=3 | 224 (223-224) n=3 | 278 (278-278) n=3 | **+13.5%** | **+9.7%** | **-11.8%** | 23.7/44.9 | 24.6/46.0 | 19.7/68.8 | 16.4/28.9 | 3.96/4.02 | 4.51/4.57 | 4.32/4.37 | 3.51/3.59 |
| 4 | 564 (563-564) n=3 | 506 (506-507) n=3 | 564 (563-565) n=3 | 633 (632-634) n=3 | **+11.4%** | **-0.0%** | **-10.9%** | 30.2/54.7 | 32.3/59.7 | 31.2/73.9 | 23.8/76.5 | 6.75/10.31 | 7.53/11.34 | 6.74/7.32 | 6.03/7.40 |
| 16 | 1,150 (1,148-1,152) n=3 | 1,095 (1,095-1,097) n=3 | - | 1,350 (1,350-1,351) n=3 | **+5.0%** | - | **-14.8%** | 47.5/364.5 | 52.1/359.6 | - | 38.0/89.4 | 13.28/24.03 | 13.95/20.11 | - | 11.37/13.41 |
| 32 | 1,777 (1,766-1,785) n=3 | 1,688 (1,685-1,692) n=3 | - | 2,087 (2,084-2,089) n=3 | **+5.2%** | - | **-14.8%** | 55.7/463.5 | 56.9/477.4 | - | 43.6/101.1 | 17.27/24.69 | 18.21/26.78 | - | 14.84/16.64 |

#### omni_t2s (text -> speech, benchmark.runner)

audio-s/s and text tok/s: mean (min-max) over rounds; TTFA (first audio) and TTFT (first text) p50/p99 and RTF p50: median over rounds of each round's value.

| c | arm | audio s/s | text tok/s | TTFA p50/p99 ms | TTFT p50/p99 ms | RTF p50 | mean audio s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mstar-pr | 20.3 (20.2-20.3) n=3 | 65 | 79/83 | 28/30 | 0.049 | 68.367 |
| 1 | mstar-main | 16.0 (15.9-16.1) n=3 | 52 | 186/194 | 27/30 | 0.062 | 67.047 |
| 1 | vllm-omni | 11.6 (11.6-11.7) n=6 | - | 88/93 | -/- | 0.086 | 79.73949999999999 |
| 1 | sglang-omni | 15.3 (15.2-15.3) n=3 | 49 | 104/146 | 18/61 | 0.065 | 68.666 |
| 4 | mstar-pr | 53.2 (52.5-54.4) n=3 | 174 | 120/143 | 30/56 | 0.074 | 70.734 |
| 4 | mstar-main | 43.0 (42.1-43.8) n=3 | 142 | 275/314 | 32/56 | 0.093 | 70.044 |
| 4 | vllm-omni | 32.9 (32.7-33.2) n=6 | - | 128/179 | -/- | 0.119 | 82.372 |
| 4 | sglang-omni | 51.5 (51.3-51.7) n=3 | 164 | 127/208 | 21/79 | 0.075 | 72.308 |
| 16 | mstar-pr | 145.2 (143.8-147.5) n=3 | 467 | 164/276 | 37/100 | 0.104 | 72.817 |
| 16 | mstar-main | 113.3 (112.0-114.7) n=3 | 364 | 390/561 | 40/96 | 0.134 | 72.351 |
| 16 | vllm-omni | 80.4 (80.2-80.6) n=6 | - | 201/416 | -/- | 0.192 | 87.5625 |
| 16 | sglang-omni | 133.4 (132.2-135.1) n=3 | 415 | 172/338 | 24/88 | 0.116 | 74.742 |
| 32 | mstar-pr | 221.8 (220.7-222.6) n=3 | 713 | 211/477 | 42/145 | 0.137 | 72.389 |
| 32 | mstar-main | 169.5 (166.1-173.0) n=3 | 548 | 504/803 | 44/144 | 0.178 | 71.836 |
| 32 | vllm-omni | 108.5 (107.5-109.0) n=6 | - | 296/542 | -/- | 0.289 | 86.536 |
| 32 | sglang-omni | 200.4 (200.0-201.2) n=3 | 624 | 231/550 | 27/102 | 0.156 | 74.725 |

Excluded cells (failed != 0 or no result):

- vllm-omni omni_t2t c=16 r1: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=16 r1: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=32 r1: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=32 r1: failed=1 ['empty output'] 
- vllm-omni omni_t2s c=1 r1: failed=0 None 
- vllm-omni omni_t2s c=4 r1: failed=0 None 
- vllm-omni omni_t2s c=16 r1: failed=0 None 
- vllm-omni omni_t2s c=32 r1: failed=0 None 
- vllm-omni omni_t2t c=16 r2: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=16 r2: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=32 r2: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=32 r2: failed=1 ['empty output'] 
- vllm-omni omni_t2s c=1 r2: failed=0 None 
- vllm-omni omni_t2s c=4 r2: failed=0 None 
- vllm-omni omni_t2s c=16 r2: failed=0 None 
- vllm-omni omni_t2s c=32 r2: failed=0 None 
- vllm-omni omni_t2t c=16 r3: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=16 r3: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=32 r3: failed=1 ['empty output'] 
- vllm-omni omni_t2t c=32 r3: failed=1 ['empty output'] 
- vllm-omni omni_t2s c=1 r3: failed=0 None 
- vllm-omni omni_t2s c=4 r3: failed=0 None 
- vllm-omni omni_t2s c=16 r3: failed=0 None 
- vllm-omni omni_t2s c=32 r3: failed=0 None 
- vllm-omni omni_t2s c=1 r1: failed=0 None 
- vllm-omni omni_t2s c=4 r1: failed=0 None 
- vllm-omni omni_t2s c=16 r1: failed=0 None 
- vllm-omni omni_t2s c=32 r1: failed=0 None 
- vllm-omni omni_t2s c=1 r2: failed=0 None 
- vllm-omni omni_t2s c=4 r2: failed=0 None 
- vllm-omni omni_t2s c=16 r2: failed=0 None 
- vllm-omni omni_t2s c=32 r2: failed=0 None 
- vllm-omni omni_t2s c=1 r3: failed=0 None 
- vllm-omni omni_t2s c=4 r3: failed=0 None 
- vllm-omni omni_t2s c=16 r3: failed=0 None 
- vllm-omni omni_t2s c=32 r3: failed=0 None 

