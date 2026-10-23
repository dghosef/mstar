Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR 30.45 dB, p10 PSNR 24.86 dB, median SSIM 0.9763; ref_a CLIP 27.94; different-seed control median PSNR 10.65 dB; competitor same-image threshold 20.55 dB (amended gate, see dgate.py)

| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | sanity gate | parity (annotated) | parity tier | original pre-registered verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| ref_b | 32.39 | 26.51 | 22.55 | 0.9836 | 27.92 | 0 | yardstick | - | - | - |
| ref_k | 32.16 | 26.04 | 20.95 | 0.9844 | 27.90 | 0 | yardstick | - | - | - |
| ref_c | 30.45 | 24.86 | 19.61 | 0.9763 | 27.93 | 0 | yardstick | - | - | - |
| ref_seed | 10.65 | 8.26 | 6.47 | 0.3536 | 27.82 | 0 | yardstick | - | - | - |
| mstar_pr | 31.77 | 25.91 | 21.53 | 0.9833 | 27.96 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_pr_b | 31.77 | 25.91 | 21.53 | 0.9833 | 27.96 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_main | 31.77 | 25.91 | 21.53 | 0.9833 | 27.95 | 0 | PASS | yes | parity (M*) | PASS |
| sglang__base | 24.37 | 20.34 | 15.51 | 0.8933 | 27.93 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__bcg | 24.37 | 20.34 | 15.51 | 0.8933 | 27.93 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__bcg_b8 | 24.37 | 20.34 | 15.51 | 0.8933 | 27.93 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__compile | 24.28 | 20.09 | 15.17 | 0.8812 | 27.91 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__compile_b8 | 24.02 | 20.13 | 15.41 | 0.8804 | 27.87 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__base | 25.40 | 21.21 | 16.29 | 0.9394 | 27.86 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__cudnn | 25.75 | 21.71 | 16.41 | 0.9421 | 27.88 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__eager | 25.28 | 21.52 | 16.34 | 0.9352 | 27.86 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__full | 25.25 | 20.83 | 16.61 | 0.9305 | 27.86 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| ref_a | - | - | - | - | 27.94 | 0 | reference | - | - | - |
