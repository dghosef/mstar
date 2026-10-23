# video agent — SUMMARY (final, 13:20 PDT)

Models: Wan2.2-TI2V-5B (`wan22`), Cosmos3 Nano / Edge, Cosmos3-Super TP2 (claimed from the pool).
Hardware: H200. 21:00-23:00 GPU1 of ovn-tts (guest); since 23:00 ovn-video (2x H200).
Branch `perf/ovn-video` (worktree /Users/josephtan/mstar-ovn-video), rebased on shared head a51da9261
(11 commits ahead; CPU tests for wan22/cosmos3/warmup/conductor/ragged: 297 pass on ovn-video).
Merged to shared from this branch: opaque ragged op (b20071aa4), least-loaded DP replica (ca5324b95),
warm every DP replica (a5fa2ec21).
Harness: results/video/harness/ovn_video_bench.py (fresh server per arm, MSTAR_PHASE_TIMING on,
per-node phases from --log-stats-file, nsys option, PSNR of decoded frames). Raw runs on
ovn-video:/workspace/results/video; text copies in results/video/runs_video/.

CPU/GPU verdict (before and after): GPU-bound. Worker phases show the step as gpu_thread.exec
(the forwards host-sync at step start) and nsys kernel time ~= wall (wan22 955 of 966 ms/step,
cosmos3 nano fully busy). Kernels run near the roofline: GEMM ~730 TF/s, FA3 ~590-680 TF/s,
the Wan VAE decode convs ~580 TF/s.

## Wan2.2-TI2V-5B (1280x704x81, guidance 5, fixed seeds)

| config | base (55b0215a3) | now |
| --- | --- | --- |
| 1 GPU, 20 steps, c=1: DiT step / e2e | 1062 ms / 24.70 s | 973 ms / 22.43 s (-9.2%) |
| 1 GPU, 50 steps, c=1: e2e | 56.95 s | 51.90 s (-8.9%) |
| 2 GPU sp2, 50 steps, c=1: e2e | (56.95 s on 1 GPU) | 27.99 s (2.03x) |
| 2 GPU dp2, 50 steps, c=4 | 1.05 videos/min (1 GPU) | 2.31 videos/min |
| 2 GPU, 20 steps, c=4: dp2 / sp2 | random replica 3.58 vpm | 5.21 / 4.69 vpm |
| 480p (832x480x81), 20 steps, c=1 | 9.33 s | 8.42 s |
| I2V 1280x704, 20 steps, c=1 | 24.56 s | 22.65 s (PSNR 36-38 dB) |
Stage breakdown now (1 GPU, 20 steps): text 0.05 s, denoise 19.5 s, VAE decode 2.34 s, mux 0.62 s.

Kept commits (each: fresh-server A/B, >=2 rounds or both GPUs crossed; output PSNR vs the previous
build inside the 29-44 dB spread a legitimate attention-kernel swap produces on this model, every
build deterministic across processes):
- shared b20071aa4 (my aa385b022, merged): `RaggedAttentionCallable(opaque=True)` custom op +
  `shard_heads` opt-out. Graph breaks around ragged attention cost ~87 ms/step of lost fusion.
- 3c410c925 self-attention on the ragged resource (FA3): step 1062 -> 1000 ms (-6%).
- 5ce473140 configs/wan22_sp2.yaml: two ranks split the CFG batch (all-gather of the noise
  prediction): step 1003 -> 502 ms. Robustness probe: mixed guidance + client abort, no hang.
- shared ca5324b95 (my 17d61232f, merged): least-loaded DP replica choice; dp2 c=4 3.58 -> 5.21 vpm.
- d58d37a89 configs/wan22_dp2.yaml.
- 2f03379d1 shipped configs compile the fast VAE decode for 832x480: decode 1.78 -> 1.04 s.
- 80f2decbb FFN gelu(tanh) in the cuBLASLt epilogue: -1.6%/step.
- bafb321ba mp4 mux with one threaded contiguous copy: mux 0.90 -> 0.70 s (sp2 1.23 -> 0.68 s), same bytes.
- 1b05dabad timestep embedded per latent frame instead of per token: -1.6%/step.

Dropped (numbers in NOTES): FA3 behind graph breaks (+0%), FA3 for the 512-token cross-attention
(0.59 vs 0.46 ms/layer), VAE latents_per_call 2/4 and cudnn.benchmark, RoPE as one stack (slower),
max-autotune compile (958 vs 912 ms), x264 frame threads (-75 ms, lower quality).

## Cosmos3 (832x480, guidance 6)
| config | base | now |
| --- | --- | --- |
| Nano 189f x 35 steps, 1 GPU, c=1 | 56.1 s (step 1.47 s, decode 2.90 s) | 55.5 s (decode 2.39 s) |
| Nano 2 GPU, c=2 | 1.07 vpm (1 GPU) | dp2 2.13 vpm (sp2 1.63) |
| Nano sp2, c=1 | 56.1 s (1 GPU) | 31.5 s (existing config) |
| Edge 121f x 20 steps, 1 GPU, c=1 | 6.58 s (decode 1.84 s) | 6.19-6.29 s (decode 1.51-1.54 s) |
| Edge sp2 (new config), c=1 | 6.27 s (1 GPU) | 4.41 s, frames identical |
| Edge dp2 (new config), c=4 | 7.9 vpm (1 GPU) | 15.8 vpm |
Kept: 28f33c8cc whole-clip VAE decode on the shared channels-last decoder (-17% decode, ~40 dB vs
stock, deterministic) + nano dp2 config; bf5af4994 tests; 46a884e7a edge sp2/dp2 configs;
[API server] one concurrent warmup copy per DP replica (cold replica's first request
10.5 -> 6.2 s; merged to shared as a5fa2ec21).
Investigated: nano sp2 "divergence" is not an SP bug (results/video/cosmos3_sp2_bug.md).

## Cosmos3-Super TP2 (claimed)
832x480x93, 20 steps: e2e 27.7-28.0 s, step 1253 ms: GEMM ~60%, FA3 24%, NCCL all-reduce 9%
(192 MB per all-reduce, 0.95 ms in-server vs 0.65 ms isolated; NCCL's default protocol is the
fastest in isolation). Async-TP probe (symm_mem fused all-gather-matmul / matmul-reduce-scatter on the
MLP) recovers 0.18 of 1.1 ms of comm per MLP, not worth a sequence-parallel rewrite. TP async sched:
no gain; NCCL_PROTO=LL128 1259 vs 1253 ms. Plateau; no change kept.

## TP async scheduling on the video lockstep nodes (12:40-13:20, coordinator priority 1)
`MSTAR_TP_ASYNC_SCHED=1`, 2x H200, fresh servers, c=1 (off -> on, DiT step):
edge sp2 124 -> 125 ms, wan22 sp2 487 -> 487 ms, cosmos3-super tp2 1253 -> 1258 ms,
nano sp2 788 -> 792 ms; edge sp2 c=4 12.05 -> 12.15 videos/min (noise). Frames bit-identical on vs off
for all four. Lockstep stress with it on (edge sp2 and wan22 sp2: 3 rounds x 6 concurrent requests,
mixed guidance 1/5/6 and steps, a third cancelled by the client mid-flight, then probes): both PASS,
no hang. Verdict: safe, but no gain for diffusion DiTs (0.12-1.3 s of GPU work per step, so the
serial schedule is <1%); not enabled in the video configs.

## Open leads
- Cosmos3 sp2 as a CFG split instead of Ulysses (edge sp2 spends ~12 ms of 126 ms/step in NCCL
  SendRecv); needs paged KV / attention resources with a shard_heads opt-out.
- TP all-reduce for Cosmos3-Super (9% of the step): in-server 0.95 vs isolated 0.65 ms per call,
  hypothesis rank skew (unverified).
- Cosmos3: FlashInfer FA3 ragged is 2-4% faster than fa3-fwd varlen at the gen-attention shapes
  (nano 17.5 vs 18.2 ms); worth ~1-2% of a step via an engine dense-ragged backend.
- ltx2_5_model.postprocess has the same permuted-view mux pattern fixed for wan22 (not mine).
- wan22 DiT step CUDA-graph capture: ~1-2% (8-11 ms idle per step).
