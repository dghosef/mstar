Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR 27.94 dB, p10 PSNR 21.41 dB, median SSIM 0.9523; ref_a CLIP 27.66; different-seed control median PSNR 10.94 dB; competitor same-image threshold 19.44 dB (amended gate, see dgate.py)

| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | tier | verdict | original verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| ref_b | 31.10 | 23.29 | 18.36 | 0.9672 | 27.61 | 0 | yardstick | - | - |
| ref_k | 28.89 | 23.84 | 19.62 | 0.9586 | 27.57 | 0 | yardstick | - | - |
| ref_c | 27.94 | 21.41 | 15.24 | 0.9523 | 27.66 | 0 | yardstick | - | - |
| ref_seed | 10.94 | 6.79 | 4.64 | 0.4235 | 27.84 | 0 | yardstick | - | - |
| mstar_pr | 29.47 | 24.20 | 17.77 | 0.9655 | 27.58 | 0 | parity (M*) | PASS | PASS |
| mstar_pr_b | 30.19 | 23.49 | 15.36 | 0.9612 | 27.69 | 0 | parity (M*) | PASS | PASS |
| mstar_main | 29.95 | 24.85 | 15.53 | 0.9623 | 27.82 | 0 | parity (M*) | PASS | PASS |
| sglang__base | 24.26 | 20.09 | 14.74 | 0.8968 | 27.98 | 0 | same-image (competitor, amended) | PASS | FAIL |
| sglang__bcg | 24.11 | 20.09 | 14.74 | 0.8960 | 27.97 | 0 | same-image (competitor, amended) | PASS | FAIL |
| sglang__bcg_b8 | 24.11 | 20.09 | 14.74 | 0.8960 | 27.97 | 0 | same-image (competitor, amended) | PASS | FAIL |
| sglang__compile | 24.31 | 20.70 | 17.67 | 0.8896 | 27.92 | 0 | same-image (competitor, amended) | PASS | FAIL |
| sglang__compile_b8 | 24.30 | 20.15 | 17.67 | 0.8924 | 27.88 | 0 | same-image (competitor, amended) | PASS | FAIL |
| vllmomni__base | 23.81 | 18.79 | 13.66 | 0.9028 | 27.96 | 0 | same-image (competitor, amended) | PASS | FAIL |
| vllmomni__cudnn | 24.01 | 18.67 | 13.97 | 0.8984 | 27.93 | 0 | same-image (competitor, amended) | PASS | FAIL |
| vllmomni__eager | 23.81 | 18.87 | 13.94 | 0.9014 | 27.83 | 0 | same-image (competitor, amended) | PASS | FAIL |
| vllmomni__full | 23.75 | 18.91 | 13.54 | 0.9034 | 28.01 | 0 | same-image (competitor, amended) | PASS | FAIL |
| ref_a | - | - | - | - | 27.66 | 0 | reference | - | - |
