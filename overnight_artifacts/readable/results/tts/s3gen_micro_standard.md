# S3Gen micro-benchmarks: standard (NVIDIA H200, torch 2.12.1+cu132)

## Kernel profile of one estimator call (2 rows, 416 frames)

**estimator fp32**: 1202 kernel launches, GPU busy 10.6 ms of 33.0 ms wall (32 % busy)

kernel | launches | GPU ms | share
--- | --- | --- | ---
`fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, ...` | 56 | 3.10 | 29 %
`sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4...` | 168 | 2.37 | 22 %
`sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x64x8_stage3_warpsize1x2x1_ffma_aligna4...` | 168 | 2.13 | 20 %
`sm80_xmma_fprop_implicit_gemm_f32f32_f32f32_f32_nchwkcrs_nchw_tilesize32x32x8_stage3_wa...` | 44 | 1.02 | 10 %
`void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float, fals...` | 141 | 0.51 | 5 %
`sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_warpsize1x4x1_ffma_aligna4...` | 112 | 0.37 | 3 %
`void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | 75 | 0.20 | 2 %
`void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | 74 | 0.17 | 2 %
`void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, s...` | 112 | 0.16 | 2 %
`void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::nati...` | 77 | 0.16 | 1 %
`void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::Te...` | 56 | 0.13 | 1 %
`void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::mi...` | 43 | 0.08 | 1 %

**estimator bf16**: 1226 kernel launches, GPU busy 3.9 ms of 39.5 ms wall (10 % busy)

kernel | launches | GPU ms | share
--- | --- | --- | ---
`cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x64_4x1x1_cga1...` | 56 | 0.65 | 16 %
`nvjet_sm90_tst_64x64_64x13_4x2_h_bz_TNT` | 168 | 0.51 | 13 %
`nvjet_sm90_tst_64x40_64x16_2x4_h_bz_bias_TNT` | 112 | 0.47 | 12 %
`void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, flo...` | 141 | 0.46 | 12 %
`nvjet_sm90_tst_128x64_64x8_4x2_h_bz_bias_TNT` | 56 | 0.24 | 6 %
`void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 74 | 0.20 | 5 %
`void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, ...` | 62 | 0.20 | 5 %
`void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 75 | 0.20 | 5 %
`void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::nati...` | 76 | 0.20 | 5 %
`sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize64x64x64_warpg...` | 31 | 0.19 | 5 %
`void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFlo...` | 112 | 0.14 | 3 %
`void at::native::vectorized_elementwise_kernel<8, at::native::GeluCUDAKernelImpl(at::Te...` | 56 | 0.12 | 3 %


## One estimator call

Rows are estimator rows (the standard model's CFG doubles the request rows; Turbo does not). `eager` is GPU-event time of a plain call, `graph` the same call replayed from a CUDA graph; `wall` is the CPU time of the eager call, so wall ≫ graph means launch-bound.

precision | rows | frames | eager ms | wall ms | graph ms | eager / graph
--- | --- | --- | --- | --- | --- | ---
fp32 | 2 | 416 | 22.43 | 22.45 | 11.18 | 2.0×
fp32 | 2 | 736 | 22.74 | 22.77 | 18.50 | 1.2×
fp32 | 4 | 416 | 22.70 | 22.73 | 16.23 | 1.4×
fp32 | 4 | 736 | 28.31 | 28.33 | 26.58 | 1.1×
fp32 | 16 | 416 | 46.89 | 46.90 | 45.34 | 1.0×
fp32 | 16 | 736 | 91.69 | 91.71 | 89.85 | 1.0×
tf32 | 2 | 416 | 23.13 | 23.16 | 7.78 | 3.0×
tf32 | 2 | 736 | 23.81 | 23.84 | 12.54 | 1.9×
tf32 | 4 | 416 | 23.63 | 23.66 | 10.14 | 2.3×
tf32 | 4 | 736 | 25.18 | 25.21 | 16.56 | 1.5×
tf32 | 16 | 416 | 28.45 | 28.46 | 25.35 | 1.1×
tf32 | 16 | 736 | 56.21 | 56.23 | 54.25 | 1.0×
bf16 | 2 | 416 | 24.52 | 24.55 | 4.41 | 5.6×
bf16 | 2 | 736 | 24.19 | 24.22 | 5.01 | 4.8×
bf16 | 4 | 416 | 24.32 | 24.35 | 4.93 | 4.9×
bf16 | 4 | 736 | 24.28 | 24.31 | 6.80 | 3.6×
bf16 | 16 | 416 | 24.22 | 24.26 | 10.48 | 2.3×
bf16 | 16 | 736 | 24.32 | 24.35 | 16.75 | 1.5×
fp16 | 2 | 416 | 24.43 | 24.46 | 4.42 | 5.5×
fp16 | 2 | 736 | 24.47 | 24.50 | 5.02 | 4.9×
fp16 | 4 | 416 | 24.43 | 24.46 | 4.91 | 5.0×
fp16 | 4 | 736 | 25.00 | 25.04 | 6.81 | 3.7×
fp16 | 16 | 416 | 24.54 | 24.56 | 10.60 | 2.3×
fp16 | 16 | 736 | 24.89 | 24.92 | 16.76 | 1.5×

## Whole flow solve per chunk (10 steps, prompt 314 frames)

`today` is `S3Gen.tokens_to_mel_rows` as served (fp32 eager, flow encoder + solve). The precision columns run the same solve with the estimator in that precision and the state in fp32 (encoder excluded); `graph` captures the whole fp32-state / bf16-estimator solve as one CUDA graph.

request rows | chunk tokens | final | frames | today ms | fp32 ms | tf32 ms | bf16 ms | fp16 ms | bf16 graph ms | Δmel tf32 | Δmel bf16 | Δmel fp16 | mel corr bf16 | wav SNR bf16 dB
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
1 | 15 | no | 344 | 239 | 231 | 236 | 249 | 249 | 43 | 0.005 | 0.099 | 0.023 | 0.9999 | 20.6
1 | 50 | no | 414 | 256 | 248 | 255 | 260 | 250 | 46 | 0.006 | 0.110 | 0.018 | 0.9999 | 16.6
8 | 50 | no | 414 | 483 | 471 | 276 | 249 | 249 | 103 | 0.087 | 0.375 | 0.176 | 0.9999 | 22.2
8 | 200 | yes | 714 | 919 | 897 | 554 | 250 | 249 | 167 | 0.026 | 1.776 | 0.142 | 0.9999 | 12.4

## Flow encoder and HiFT vocoder

stage | rows | tokens or mel frames | eager ms | wall ms
--- | --- | --- | --- | ---
flow encoder | 1 | 172 | 6.94 | 6.97
flow encoder | 1 | 207 | 6.91 | 6.94
flow encoder | 1 | 357 | 6.94 | 6.97
flow encoder | 8 | 172 | 8.69 | 8.70
flow encoder | 8 | 207 | 9.92 | 9.93
flow encoder | 8 | 357 | 18.01 | 18.02
HiFT vocode | 1 | 38 | 11.32 | 11.34
HiFT vocode | 1 | 108 | 11.37 | 11.40
HiFT vocode | 1 | 208 | 11.31 | 11.34
HiFT vocode | 1 | 408 | 14.33 | 14.35
HiFT vocode | 1 | 800 | 24.43 | 24.45

HiFT draws its excitation noise inside `vocode`, so a graph capture would have to take the noise as an input; the flow encoder's length varies per chunk (bucketing needed).

## Watermark

resemble-perth is not installed

