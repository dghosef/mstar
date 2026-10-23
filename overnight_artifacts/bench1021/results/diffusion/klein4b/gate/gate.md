Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR 32.95 dB, p10 PSNR 26.40 dB, median SSIM 0.9847; ref_a CLIP 27.62; different-seed control median PSNR 10.46 dB; competitor same-image threshold 21.71 dB (amended gate, see dgate.py)

| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | sanity gate | parity (annotated) | parity tier | original pre-registered verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| ref_b | 34.38 | 29.02 | 24.71 | 0.9903 | 27.61 | 0 | yardstick | - | - | - |
| ref_k | 34.50 | 28.85 | 23.35 | 0.9885 | 27.61 | 0 | yardstick | - | - | - |
| ref_c | 32.95 | 26.40 | 18.54 | 0.9847 | 27.54 | 0 | yardstick | - | - | - |
| ref_seed | 10.46 | 8.56 | 6.47 | 0.3942 | 27.37 | 0 | yardstick | - | - | - |
| mstar_pr | 34.98 | 28.57 | 25.70 | 0.9902 | 27.61 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_pr_b | 34.98 | 28.57 | 25.70 | 0.9902 | 27.61 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_main | 34.99 | 28.57 | 25.70 | 0.9902 | 27.60 | 0 | PASS | yes | parity (M*) | PASS |
| sglang__base | 22.46 | 17.48 | 14.32 | 0.8666 | 27.56 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__bcg | 22.46 | 17.48 | 14.32 | 0.8666 | 27.56 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__bcg_b8 | 22.46 | 17.48 | 14.32 | 0.8666 | 27.56 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__compile | 22.20 | 17.21 | 14.38 | 0.8652 | 27.49 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__compile_b8 | 22.23 | 17.31 | 14.71 | 0.8674 | 27.53 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__base | 23.03 | 18.59 | 15.46 | 0.9176 | 27.59 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__cudnn | 23.35 | 18.65 | 15.49 | 0.9219 | 27.60 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__eager | 23.22 | 18.59 | 15.37 | 0.9221 | 27.63 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__full | 22.94 | 18.40 | 15.85 | 0.9189 | 27.62 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| ref_a | - | - | - | - | 27.62 | 0 | reference | - | - | - |
