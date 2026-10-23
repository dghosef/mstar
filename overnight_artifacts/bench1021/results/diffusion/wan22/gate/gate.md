Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR 28.52 dB, p10 PSNR 22.10 dB, median SSIM 0.9306; ref_a CLIP 21.28; different-seed control median PSNR 9.62 dB; competitor same-image threshold 19.07 dB (amended gate, see dgate.py)

| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | sanity gate | parity (annotated) | parity tier | original pre-registered verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| ref_b | 41.16 | 36.01 | 34.08 | 0.9791 | 21.34 | 0 | yardstick | - | - | - |
| ref_k | 30.62 | 23.85 | 21.16 | 0.9547 | 21.38 | 0 | yardstick | - | - | - |
| ref_c | 28.52 | 22.10 | 20.44 | 0.9306 | 21.40 | 0 | yardstick | - | - | - |
| ref_seed | 9.62 | 7.94 | 7.63 | 0.4598 | 21.32 | 0 | yardstick | - | - | - |
| mstar_pr | 28.09 | 20.42 | 18.05 | 0.9309 | 21.60 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_pr_b | 28.09 | 20.42 | 18.05 | 0.9309 | 21.60 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_main | 28.67 | 20.74 | 18.03 | 0.9357 | 21.65 | 0 | PASS | yes | parity (M*) | PASS |
| sglang__base | 27.04 | 23.18 | 22.76 | 0.9183 | 21.47 | 0 | PASS | yes | same-image (competitor, amended) | PASS |
| sglang__bcg | 27.04 | 23.18 | 22.76 | 0.9183 | 21.47 | 0 | PASS | yes | same-image (competitor, amended) | PASS |
| sglang__compile | 27.09 | 23.72 | 23.46 | 0.9194 | 22.16 | 0 | PASS | yes | same-image (competitor, amended) | PASS |
| sglang__compile_b4 | 27.09 | 23.72 | 23.46 | 0.9194 | 22.16 | 0 | PASS | yes | same-image (competitor, amended) | PASS |
| vllmomni__b4 | 11.65 | 8.60 | 7.35 | 0.4927 | 23.43 | 0 | PASS | no (annotated) | same-image (competitor, amended) | PASS (distribution-level only) |
| vllmomni__base | 11.65 | 8.60 | 7.35 | 0.4927 | 23.43 | 0 | PASS | no (annotated) | same-image (competitor, amended) | PASS (distribution-level only) |
| vllmomni__cudnn | 11.67 | 8.62 | 7.35 | 0.4940 | 22.82 | 0 | PASS | no (annotated) | same-image (competitor, amended) | PASS (distribution-level only) |
| vllmomni__full | 11.73 | 8.61 | 7.34 | 0.4944 | 22.81 | 0 | PASS | no (annotated) | same-image (competitor, amended) | PASS (distribution-level only) |
| ref_a | - | - | - | - | 21.28 | 0 | reference | - | - | - |
