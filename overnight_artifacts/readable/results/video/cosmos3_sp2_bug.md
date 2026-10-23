# Cosmos3-Nano sp2 vs 1-GPU divergence: not an SP bug (kernel-noise amplification)

Question (coordinator, from my NOTES 01:40): cosmos3_nano_sp2.yaml output at 832x480x189, 35 steps,
guidance 6 differs strongly from the 1-GPU output (PSNR ~20 dB; on prompt 0 a different person is
in frame). Is the Ulysses SP path wrong?

Method: same base tree (perf/overnight-shared 55b0215a3 code for cosmos3), 2x H200, fixed seeds
(seed = prompt index), PSNR of decoded frames per clip. The yardstick is a *single-GPU* run whose
only change is the attention kernel: `model_kwargs.attention_backend: flashinfer` (paged FlashInfer)
instead of the default `dense_gen` (FA3 varlen). That is a legitimate kernel swap with no SP at all.

| run | clip 0 | clip 1 |
| --- | --- | --- |
| 2 steps, 33 frames, gs 6: 1-GPU vs flashinfer-kernel 1-GPU | 34.9 dB | 41.4 dB |
| 2 steps, 33 frames, gs 6: 1-GPU vs sp2 | 35.1 dB | 41.2 dB |
| 2 steps, 189 frames, gs 6: 1-GPU vs flashinfer-kernel 1-GPU | 35.1 dB | 38.0 dB |
| 2 steps, 189 frames, gs 6: 1-GPU vs sp2 | 36.5 dB | 38.6 dB |
| 2 steps, 33 frames, gs 1: 1-GPU vs sp2 | 43.6 dB | 46.4 dB |
| 35 steps, 189 frames, gs 1: 1-GPU vs sp2 | 36.9 dB | 35.7 dB |
| 35 steps, 189 frames, gs 6: 1-GPU vs flashinfer-kernel 1-GPU | 24.3 dB | 19.7 dB |
| 35 steps, 189 frames, gs 6: 1-GPU vs sp2 | 21.6 dB | 20.1 dB |

Reading:
- After two steps, sp2 is exactly as close to the 1-GPU output as a single-GPU attention-kernel swap
  (35-41 dB both). A layout/RoPE/padding/rank-token bug would show up at step 1-2 as a gross error
  (single-digit to low-teens dB), not as kernel-level noise.
- Without guidance (gs 1) the full 35-step run stays at ~36 dB; with gs 6 *both* the kernel swap and
  sp2 end at ~20 dB. Guidance 6 x 35 steps x 189 frames amplifies bf16 kernel-level differences into a
  different sample; on prompt 0 the kernel swap keeps the person and sp2 lands on another one, on prompt
  1 sp2 is visually closer than the kernel swap (frames: results/video/frames/c_cmp_fi*.jpg).
- Same pattern documented for batching in docs/models.rst ("~30 dB PSNR at guidance 6").

Verdict: no SP correctness bug found. Ulysses sp2 output is within the run-to-run spread of valid
kernels for this model at guidance 6. The real (pre-existing) property is that Cosmos3 video at gs 6
is not reproducible across *any* numerics change (kernel, batching, SP); bitwise repeatability needs
same build, same config, one request at a time. Determinism within a config holds: repeated
processes give identical frames (PSNR inf).

Runs: /workspace/results/video/{cs_*,cs189_*,c_1g_fi,c_1g_full_gs1,c_sp2_full_gs1,c_base_c1,c_base_sp2_c1}
on ovn-video.
