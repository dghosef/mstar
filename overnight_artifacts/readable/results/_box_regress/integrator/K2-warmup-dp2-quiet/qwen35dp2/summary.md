## qwen35dp2 (GPU 2,3)  base=7fd1e3589 perf/overnight-shared dirty=0  cand=902ed51b4 perf/ovn-integ-cand-k dirty=0
| metric | base (rounds) | cand (rounds) | delta |
| --- | --- | --- | --- |
| c1_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c1_itl_ms | 1.66, 1.472, 1.42, 1.472 | 1.389, 1.465, 1.435, 1.595 | -2.33% |
| c1_tok_s | 582.6, 661.6, 684.7, 661 | 699.1, 664.9, 677.9, 609.9 | +2.39% |
| c1_ttft_ms | 13.39, 10.97, 11.7, 11.4 | 11.75, 11.5, 11.92, 12.62 | +0.71% |
| c32_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c32_itl_ms | 1.874, 1.96, 1.872, 1.889 | 1.828, 1.802, 1.957, 2.13 | +1.61% |
| c32_tok_s | 1.534e+04, 1.464e+04, 1.561e+04, 1.541e+04 | 1.584e+04, 1.595e+04, 1.468e+04, 1.364e+04 | -1.47% |
| c32_ttft_p50_ms | 52.43, 49.54, 46.87, 49.63 | 47.4, 44.88, 45.21, 52.21 | -4.42% |
| c64_failed | 0, 0, 0, 0 | 0, 0, 0, 0 |  |
| c64_itl_ms | 3.613, 3.686, 3.845, 3.964 | 3.792, 3.802, 3.92, 4.098 | +3.34% |
| c64_tok_s | 1.629e+04, 1.59e+04, 1.518e+04, 1.483e+04 | 1.492e+04, 1.557e+04, 1.5e+04, 1.441e+04 | -3.70% |
| c64_ttft_p50_ms | 74.92, 81.84, 82.57, 86.11 | 95.6, 76.92, 82.41, 83.82 | +4.09% |

Correctness: {"base_vs_cand_match": 6, "n": 8, "base_r0_vs_r1_match": 6, "cand_r0_vs_r1_match": 6, "cand_outputs_seen_in_base": 8, "first_diff": {"prompt": 3, "common_prefix_chars": 269, "base": ")\n    Wait, haiku is 5-7-5 syllables.\n    Let's try to get the rhythm right.\n\n  ", "cand": " No, haiku is 3 lines.)\n    Wait, haiku is 5-7-5 syllables.\n    Let's try to cou"}}
