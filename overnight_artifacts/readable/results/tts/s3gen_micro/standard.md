# S3Gen micro-benchmarks: standard (NVIDIA H200, torch 2.12.1+cu132)

## Kernel profile of one estimator call (2 rows, 416 frames)

**estimator fp32**: 1202 kernel launches, GPU busy 10.6 ms of 35.5 ms wall (30 % busy)

kernel | launches | GPU ms | share
--- | --- | --- | ---
`fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, ...` | 56 | 3.12 | 29 %
`sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4...` | 168 | 2.36 | 22 %
`sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x64x8_stage3_warpsize1x2x1_ffma_aligna4...` | 168 | 2.13 | 20 %
`sm80_xmma_fprop_implicit_gemm_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize32x32x8_stage3_wa...` | 44 | 1.01 | 10 %
`void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, fals...` | 141 | 0.51 | 5 %
`sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4...` | 112 | 0.37 | 3 %
`void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | 75 | 0.20 | 2 %
`void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | 74 | 0.17 | 2 %
`void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, s...` | 112 | 0.16 | 2 %
`void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | 77 | 0.16 | 1 %
`void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::Te...` | 56 | 0.13 | 1 %
`void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::mi...` | 43 | 0.08 | 1 %

**estimator bf16**: 1226 kernel launches, GPU busy 3.9 ms of 36.5 ms wall (11 % busy)

kernel | launches | GPU ms | share
--- | --- | --- | ---
`cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x64_4x1x1_cga1...` | 56 | 0.65 | 16 %
`nvjet_sm90_tst_64x64_64x13_4x2_h_bz_TNT` | 168 | 0.51 | 13 %
`nvjet_sm90_tst_64x40_64x16_2x4_h_bz_bias_TNT` | 112 | 0.47 | 12 %
`void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, flo...` | 141 | 0.46 | 12 %
`nvjet_sm90_tst_128x64_64x8_4x2_h_bz_bias_TNT` | 56 | 0.23 | 6 %
`void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 74 | 0.20 | 5 %
`void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 75 | 0.20 | 5 %
`void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, ...` | 62 | 0.20 | 5 %
`void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 76 | 0.20 | 5 %
`sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_warpg...` | 31 | 0.19 | 5 %
`void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFlo...` | 112 | 0.14 | 3 %
`void at::native::vectorized_elementwise_kernel<8, at::native::GeluCUDAKernelImpl(at::Te...` | 56 | 0.12 | 3 %


## One estimator call

Rows are estimator rows (the standard model's CFG doubles the request rows; Turbo does not). `eager` is GPU-event time of a plain call, `graph` the same call replayed from a CUDA graph; `wall` is the CPU time of the eager call, so wall ≫ graph means launch-bound.

precision | rows | frames | eager ms | wall ms | graph ms | eager / graph
--- | --- | --- | --- | --- | --- | ---
fp32 | 1 | 352 | 23.14 | 23.16 | 9.25 | 2.5×
fp32 | 1 | 416 | 22.20 | 22.22 | 9.82 | 2.3×
fp32 | 1 | 512 | 23.55 | 23.58 | 10.41 | 2.3×
fp32 | 1 | 736 | 22.04 | 22.07 | 12.91 | 1.7×
fp32 | 1 | 1024 | 23.47 | 23.50 | 15.28 | 1.5×
fp32 | 2 | 352 | 22.46 | 22.49 | 10.38 | 2.2×
fp32 | 2 | 416 | 23.00 | 23.03 | 11.15 | 2.1×
fp32 | 2 | 512 | 24.30 | 24.34 | 12.03 | 2.0×
fp32 | 2 | 736 | 22.21 | 22.24 | 18.51 | 1.2×
fp32 | 2 | 1024 | 24.36 | 24.37 | 22.59 | 1.1×
fp32 | 4 | 352 | 23.09 | 23.12 | 14.33 | 1.6×
fp32 | 4 | 416 | 22.36 | 22.39 | 15.91 | 1.4×
fp32 | 4 | 512 | 28.84 | 28.87 | 17.73 | 1.6×
fp32 | 4 | 736 | 28.57 | 28.59 | 26.63 | 1.1×
fp32 | 4 | 1024 | 41.52 | 41.54 | 39.47 | 1.1×
fp32 | 8 | 352 | 23.34 | 23.35 | 21.54 | 1.1×
fp32 | 8 | 416 | 29.14 | 29.15 | 27.23 | 1.1×
fp32 | 8 | 512 | 32.15 | 32.16 | 30.23 | 1.1×
fp32 | 8 | 736 | 49.37 | 49.39 | 47.78 | 1.0×
fp32 | 8 | 1024 | 68.29 | 68.31 | 66.81 | 1.0×
fp32 | 16 | 352 | 38.60 | 38.61 | 36.86 | 1.0×
fp32 | 16 | 416 | 47.28 | 47.30 | 45.50 | 1.0×
fp32 | 16 | 512 | 52.29 | 52.31 | 50.74 | 1.0×
fp32 | 16 | 736 | 91.97 | 92.01 | 89.88 | 1.0×
fp32 | 16 | 1024 | 130.18 | 130.20 | 127.91 | 1.0×
tf32 | 1 | 352 | 21.74 | 21.77 | 7.18 | 3.0×
tf32 | 1 | 416 | 24.38 | 24.42 | 7.60 | 3.2×
tf32 | 1 | 512 | 23.10 | 23.14 | 8.24 | 2.8×
tf32 | 1 | 736 | 23.91 | 23.96 | 9.76 | 2.5×
tf32 | 1 | 1024 | 22.93 | 22.96 | 11.69 | 2.0×
tf32 | 2 | 352 | 24.44 | 24.47 | 7.25 | 3.4×
tf32 | 2 | 416 | 23.27 | 23.30 | 7.78 | 3.0×
tf32 | 2 | 512 | 23.56 | 23.60 | 8.35 | 2.8×
tf32 | 2 | 736 | 24.20 | 24.24 | 12.57 | 1.9×
tf32 | 2 | 1024 | 25.23 | 25.26 | 15.42 | 1.6×
tf32 | 4 | 352 | 25.12 | 25.16 | 9.03 | 2.8×
tf32 | 4 | 416 | 23.40 | 23.43 | 10.01 | 2.3×
tf32 | 4 | 512 | 24.28 | 24.31 | 10.87 | 2.2×
tf32 | 4 | 736 | 25.20 | 25.25 | 16.52 | 1.5×
tf32 | 4 | 1024 | 29.14 | 29.16 | 27.06 | 1.1×
tf32 | 8 | 352 | 24.53 | 24.56 | 11.73 | 2.1×
tf32 | 8 | 416 | 25.46 | 25.49 | 15.70 | 1.6×
tf32 | 8 | 512 | 24.96 | 24.99 | 18.01 | 1.4×
tf32 | 8 | 736 | 31.28 | 31.29 | 29.09 | 1.1×
tf32 | 8 | 1024 | 48.08 | 48.11 | 45.77 | 1.1×
tf32 | 16 | 352 | 26.11 | 26.14 | 20.21 | 1.3×
tf32 | 16 | 416 | 27.72 | 27.73 | 25.25 | 1.1×
tf32 | 16 | 512 | 31.73 | 31.74 | 29.58 | 1.1×
tf32 | 16 | 736 | 55.86 | 55.87 | 53.93 | 1.0×
tf32 | 16 | 1024 | 89.37 | 89.39 | 87.50 | 1.0×
bf16 | 1 | 352 | 23.64 | 23.66 | 4.03 | 5.9×
bf16 | 1 | 416 | 28.21 | 28.24 | 4.18 | 6.7×
bf16 | 1 | 512 | 23.82 | 23.85 | 4.20 | 5.7×
bf16 | 1 | 736 | 26.24 | 26.27 | 4.56 | 5.8×
bf16 | 1 | 1024 | 24.73 | 24.76 | 4.85 | 5.1×
bf16 | 2 | 352 | 27.53 | 27.56 | 4.25 | 6.5×
bf16 | 2 | 416 | 24.61 | 24.64 | 4.40 | 5.6×
bf16 | 2 | 512 | 25.40 | 25.43 | 4.46 | 5.7×
bf16 | 2 | 736 | 26.50 | 26.53 | 5.01 | 5.3×
bf16 | 2 | 1024 | 30.19 | 30.23 | 5.52 | 5.5×
bf16 | 4 | 352 | 29.52 | 29.56 | 4.65 | 6.4×
bf16 | 4 | 416 | 25.72 | 25.75 | 4.93 | 5.2×
bf16 | 4 | 512 | 25.82 | 25.85 | 5.11 | 5.1×
bf16 | 4 | 736 | 27.66 | 27.69 | 6.82 | 4.1×
bf16 | 4 | 1024 | 27.94 | 27.96 | 7.97 | 3.5×
bf16 | 8 | 352 | 25.77 | 25.80 | 6.23 | 4.1×
bf16 | 8 | 416 | 28.00 | 28.03 | 6.70 | 4.2×
bf16 | 8 | 512 | 24.16 | 24.19 | 7.15 | 3.4×
bf16 | 8 | 736 | 24.58 | 24.61 | 9.92 | 2.5×
bf16 | 8 | 1024 | 24.10 | 24.13 | 12.82 | 1.9×
bf16 | 16 | 352 | 24.00 | 24.03 | 8.89 | 2.7×
bf16 | 16 | 416 | 24.04 | 24.08 | 10.47 | 2.3×
bf16 | 16 | 512 | 24.12 | 24.15 | 11.23 | 2.1×
bf16 | 16 | 736 | 24.76 | 24.79 | 16.79 | 1.5×
bf16 | 16 | 1024 | 24.65 | 24.66 | 22.30 | 1.1×
fp16 | 1 | 352 | 25.17 | 25.22 | 4.02 | 6.3×
fp16 | 1 | 416 | 24.21 | 24.24 | 4.18 | 5.8×
fp16 | 1 | 512 | 27.75 | 27.79 | 4.20 | 6.6×
fp16 | 1 | 736 | 23.85 | 23.88 | 4.56 | 5.2×
fp16 | 1 | 1024 | 24.35 | 24.38 | 4.86 | 5.0×
fp16 | 2 | 352 | 24.41 | 24.45 | 4.25 | 5.7×
fp16 | 2 | 416 | 24.03 | 24.06 | 4.41 | 5.4×
fp16 | 2 | 512 | 24.05 | 24.08 | 4.46 | 5.4×
fp16 | 2 | 736 | 23.60 | 23.63 | 5.01 | 4.7×
fp16 | 2 | 1024 | 24.01 | 24.04 | 5.50 | 4.4×
fp16 | 4 | 352 | 23.92 | 23.95 | 4.66 | 5.1×
fp16 | 4 | 416 | 24.42 | 24.44 | 4.90 | 5.0×
fp16 | 4 | 512 | 24.08 | 24.11 | 5.10 | 4.7×
fp16 | 4 | 736 | 23.85 | 23.88 | 6.80 | 3.5×
fp16 | 4 | 1024 | 23.95 | 23.98 | 7.95 | 3.0×
fp16 | 8 | 352 | 24.02 | 24.05 | 6.23 | 3.9×
fp16 | 8 | 416 | 31.04 | 31.10 | 6.73 | 4.6×
fp16 | 8 | 512 | 23.98 | 24.01 | 7.14 | 3.4×
fp16 | 8 | 736 | 24.01 | 24.04 | 9.93 | 2.4×
fp16 | 8 | 1024 | 24.10 | 24.13 | 12.81 | 1.9×
fp16 | 16 | 352 | 23.86 | 23.89 | 8.86 | 2.7×
fp16 | 16 | 416 | 24.08 | 24.11 | 10.60 | 2.3×
fp16 | 16 | 512 | 24.48 | 24.51 | 11.21 | 2.2×
fp16 | 16 | 736 | 24.37 | 24.40 | 16.77 | 1.5×
fp16 | 16 | 1024 | 24.51 | 24.52 | 22.66 | 1.1×

## Whole flow solve per chunk (10 steps, prompt 314 frames)

`today` is `S3Gen.tokens_to_mel_rows` as served (fp32 eager, flow encoder + solve). The precision columns run the same solve with the estimator in that precision and the state in fp32 (encoder excluded); `graph` captures the whole fp32-state / bf16-estimator solve as one CUDA graph.

request rows | chunk tokens | final | frames | today ms | fp32 ms | tf32 ms | bf16 ms | fp16 ms | bf16 graph ms | Δmel tf32 | Δmel bf16 | Δmel fp16 | mel corr bf16 | wav SNR bf16 dB
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
1 | 15 | no | 344 | 235 | 227 | 244 | 248 | 253 | 43 | 0.005 | 0.099 | 0.023 | 0.9999 | 20.6
1 | 50 | no | 414 | 260 | 258 | 259 | 265 | 262 | 46 | 0.006 | 0.110 | 0.018 | 0.9999 | 16.6
1 | 200 | yes | 714 | 261 | 254 | 263 | 249 | 249 | 51 | 0.016 | 0.223 | 0.060 | 0.9999 | 11.3
4 | 50 | no | 414 | 301 | 290 | 275 | 257 | 251 | 67 | 0.008 | 0.188 | 0.018 | 0.9999 | 21.1
8 | 50 | no | 414 | 483 | 471 | 281 | 255 | 258 | 103 | 0.087 | 0.375 | 0.176 | 0.9999 | 22.2
8 | 200 | no | 714 | 914 | 894 | 553 | 249 | 254 | 167 | 0.055 | 0.543 | 0.079 | 0.9999 | 13.6
8 | 200 | yes | 714 | 914 | 894 | 554 | 257 | 256 | 167 | 0.026 | 1.776 | 0.142 | 0.9999 | 12.4

## Flow encoder and HiFT vocoder

stage | rows | tokens or mel frames | eager ms | wall ms
--- | --- | --- | --- | ---
flow encoder | 1 | 172 | 7.35 | 7.37
flow encoder | 1 | 207 | 6.97 | 6.99
flow encoder | 1 | 357 | 6.91 | 6.93
flow encoder | 8 | 172 | 8.68 | 8.69
flow encoder | 8 | 207 | 9.87 | 9.88
flow encoder | 8 | 357 | 17.97 | 17.98
HiFT vocode | 1 | 38 | 12.46 | 12.48
HiFT vocode | 1 | 108 | 12.09 | 12.11
HiFT vocode | 1 | 208 | 11.24 | 11.26
HiFT vocode | 1 | 408 | 14.29 | 14.31
HiFT vocode | 1 | 800 | 24.40 | 24.43

HiFT draws its excitation noise inside `vocode`, so a graph capture would have to take the noise as an input; the flow encoder's length varies per chunk (bucketing needed).

## Watermark

resemble-perth is not installed

