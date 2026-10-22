## qwen35dp2 (GPU 2,3)  base=7fd1e3589 perf/overnight-shared dirty=0  cand=65ca33950 perf/ovn-integ-cand-k dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0 | 0, 0 |  |
| c1_itl_ms | 1.565, 1.387 | 1.555, 1.446 | +1.66% |
| c1_tok_s | 622.1, 701.7 | 625.5, 670.2 | -2.12% |
| c1_ttft_ms | 12.39, 10.66 | 12.74, 13.23 | +12.68% |
| c32_failed | 0, 0 | 0, 0 |  |
| c32_itl_ms | 1.932, 1.816 | 2.084, 2.019 | +9.47% |
| c32_tok_s | 1.452e+04, 1.597e+04 | 1.377e+04, 1.432e+04 | -7.86% |
| c32_ttft_p50_ms | 48.47, 45.45 | 57.64, 49.45 | +14.01% |
| c64_failed | 0, 0 | 0, 0 |  |
| c64_itl_ms | 3.706, 3.366 | 3.959, 3.947 | +11.79% |
| c64_tok_s | 1.578e+04, 1.737e+04 | 1.471e+04, 1.505e+04 | -10.23% |
| c64_ttft_p50_ms | 82.73, 80.69 | 88.29, 75.16 | +0.02% |

Correctness: {"base_vs_cand_match": 7, "n": 8, "base_r0_vs_r1_match": 6, "cand_r0_vs_r1_match": 7, "cand_outputs_seen_in_base": 8, "first_diff": {"prompt": 7, "common_prefix_chars": 374, "base": "The marriage of the two families.\n    *   Climax: The night of the wedding.\n    ", "cand": "Romeo and Juliet meet in the garden.\n    *   Climax: The night of the wedding.\n "}}
