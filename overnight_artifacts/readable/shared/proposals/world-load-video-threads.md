# world: decode request videos with 8 FFmpeg threads (shared Model.load_video)

- Branch `perf/ovn-world`, commit `7d523fbf6` (on top of `cebd8b4a1`, the V-JEPA2-only version it generalizes).
  Files: `mstar/model/media_limits.py` (new `VIDEO_DECODE_THREADS = min(8, os.cpu_count())`),
  `mstar/model/base.py` (`Model.load_video` passes `num_ffmpeg_threads=VIDEO_DECODE_THREADS`),
  `mstar/model/vjepa2/vjepa2_model.py` (uses the same constant).
- Mechanism: torchcodec's `VideoDecoder` defaults to `num_ffmpeg_threads=1`, so every request video
  is decoded single-threaded in the API server before the request is even submitted. FFmpeg frame
  threading returns bit-identical frames (checked `torch.equal`).
- Evidence (H200 box ovn-world, idle CPU, `test/qwen3-omni/video.webm`, 829 frames 720p AV1):
  - decode-all-frames (the `Model.load_video` path): 3.02/3.08 s -> 0.88/1.08 s, frames identical.
  - sampled 64-frame decode (V-JEPA2 path): 1.02 s -> 0.37 s (1/2/4/8/16/0 threads: 1.02/0.63/0.47/0.36/0.36/0.48 s).
  - Served V-JEPA2 (ViT-L), 2 rounds ABBA, same box: c=1 JCT p50 2.70 -> 2.01 s; c=4 0.624 -> 0.796 req/s;
    outputs byte-identical. Results: `~/runs/overnight/results/world/` (vjepa2_ab1).
- Who it affects: every model on the default `Model.load_video` — Qwen3.5 video_to_text (qwen35vl's
  video path), Qwen3-Omni video input (it overrides `load_video`; check whether its override still
  constructs `VideoDecoder` without threads: `mstar/model/qwen3_omni/qwen3_omni_model.py:1022`),
  BAGEL/cosmos3 if they take video (cosmos3 overrides too). Text/image/audio paths untouched.
- Risk: more CPU threads per decode in the API server at high concurrency (8 per in-flight decode).
  Worth a c=16/32 video_to_text check for contention; the cap is the knob if it regresses.
- Tests: `ruff check` clean; no CPU tests touch the thread count.

## Verdict (integrator 00:20): MERGED into perf/overnight-shared as b9ae42315 + 6e9b686b5

- Applied together with its V-JEPA2 predecessor cebd8b4a1, which 7d523fbf6 builds on (the vjepa2 hunk would not apply alone).
- Review: one constant, `min(8, cpu_count)`, plus `num_ffmpeg_threads` on the default `Model.load_video` and V-JEPA2's decoder. The contention risk the proposal flags doesn't arise. Request media is decoded on the API server's single preprocess thread (`PreprocessWorkerThread`, one `threading.Thread` in data_worker.py), so at most one decode runs at a time and the cap bounds the host at 8 FFmpeg threads whatever the concurrency. That thread also streams results, so a shorter decode also shortens a stall in output streaming for other requests. No env var or global torch setting changes; text, image and audio paths are untouched.
- Independent check on ovn-regress (H200 box CPU), test/qwen3-omni/video.webm (829 frames, 1280x720), the full-decode path: 1 thread 2.09/2.09 s -> 8 threads 0.87/0.79 s, frames `torch.equal` to the single-thread decode in both repeats.
- CPU tests on the combined head (omni + qwen35vl + video + this) vs base 55b0215a3: no new failures. 3 vs 7 failures; the difference is the load-dependent test_eval_compare flakes.
- No GPU suite cell: none of the suite models decodes video on its default config (qwen3_5_0.8b.yaml refuses video). Served evidence is the proposer's V-JEPA2 A/B (c=1 JCT p50 2.70 -> 2.01 s). **qwen35vl / omni: please re-measure your video_to_text paths on the new shared head.** Qwen3-Omni overrides load_video and is not covered by this change.
