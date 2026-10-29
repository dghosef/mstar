# Qwen3-TTS 1.7B CustomVoice baseline profile (lead) — 2026-10-29 ~05:00Z

Base: perf/ovn2-final 4e3de8eae ("perf stack only" reference; a re-profile on qwen3tts/base follows when BASE.md lands).
Config configs/qwen3tts_1p7b.yaml (Talker + Codec on one worker, GPU 0). H200 (Eras 1/4 p5en slice), torch 2.11.0+cu130, FlashInfer 0.7.2.post1, SHM arena.
Workload: Seed-TTS en (1,088 sentences, ~4.7 s audio ≈ 59 frames each), closed loop, voice ryan, temp 0.9 / top-k 50 / top-p 1.0 / rep 1.05.
Evidence root: ~/runs/qwen3tts-gap-20261028-2016/results/prof/ — `base_t1/` (MSTAR_PHASE_TIMING=100 tables per c), `base_nsys_graph/` (nsys with CUDA graphs traced as whole replays: honest GPU busy/idle), `base_nsys/` (nsys with graphs traced per node: kernel composition; node tracing inflates gaps, so take busy % from base_nsys_graph). Analyzer: ~/runs/qwen3tts-gap-20261028-2016/harness/analyze_nsys.py (kernels attributed to the worker's NVTX node/walk range; capture via a cudaProfilerApi hook, harness/nsys_prof.py + harness/nsys_hook/).

## Throughput (closed loop, base_t1)

| c | audio-s/s | TTFA p50 / p99 (ms) |
|---|---:|---:|
| 1 | 16.5 | 15 / 19 |
| 16 | 120 | 35 / 71 |
| 32 | 184 | 42 / 106 |
| 64 | 246 | 56 / 198 |

TTFA has a lot of headroom against the 200 ms SLO at every c: latency can be traded for batching.

## CPU- vs GPU-bound

GPU busy (union of kernels + graph replays + memcpy, graph-level capture): c=1 94%, c=16 89%, c=32 85%, c=64 78%.
Mostly GPU-bound; idle grows with c (22% at c=64), and most of it is inside codec and prefill steps (eager glue between launches), not between steps.
Phase tables agree: c=64 iter_total 6.0 ms, event_sync 0.56 ms mean, but gpu_thread.prepare_inputs 1.95 ms mean / 8.3 ms p95 — the codec prepare_inputs' EOS boolean index (`DeviceSelectSweepKernel` in the trace) is a device sync that blocks the GPU thread; see FINDINGS (#886 port → host).

## Where GPU time goes (base_nsys_graph, ~3.6-3.9 s windows)

| c | decode: share, ms/step, steps/s | codec: share, ms/step, steps/s | prefill: share, ms/step, steps/s |
|---|---|---|---|
| 1 | 92%, 4.17, 208 | 5%, 2.27, 21 | 2.5%, 6.17, 4 |
| 16 | 60%, 5.39, 100 | 26%, 2.93, 79 | 13%, 6.22, 18 |
| 32 | 47%, 5.46, 74 | 36%, 3.38, 90 | 17%, 6.27, 23 |
| 64 | 38%, 6.14, 48 | 46%, 4.29, 83 | 16%, 6.39, 20 |

c=64 per second of wall: decode 29.5%, codec 35.6%, prefill 12.5%, idle 22% (≈6% gaps inside codec steps — ~13 launches × 54 µs around each graph replay; ≈4% inside prefill steps — ~31 gaps × 67 µs; ≈2% inside/between decode steps). Decode runs at ~64 rows (48 steps/s × 64 ≈ the 3,075 frames/s that 246 audio-s/s needs).

## Bottlenecks, ranked (c=64 unless noted)

1. **Codec batches ~4 rows per step (codec).** 83 codec steps/s against ~52 req/s × ~6 windows/request (ramp 1,3,8,16 then 25-frame windows; a 59-frame sentence is 1,3,8,16,25,6) ≈ 310 windows/s → ≈3.8 rows per step, 4.3 ms a step, 1.7 codec steps per decode step. A codec step costs ~2.3 ms at 1 row (c=1) and little more per row, so the step count, not the row count, is the cost: 35.6% of wall. Causes in code: `split_batches_by_capture_key` keeps every window bucket apart (~6 buckets, each sees well under one window per Talker step), and ChunkBatchHold(min_chunks=32, max_wait=40 ms) waits only ~2 windows' worth per bucket, while a full window behind a 2.2 s ramp has seconds of playback slack. Playback-deadline-aware release + cross-bucket batching is the largest single lever (back-of-envelope: 310 windows/s in batches of ~32 ≈ 10 steps/s ≈ 80 ms/s instead of ~360 ms/s). Row counts are estimated from window counts, not measured per step.
2. **Codec step cost (codec).** Kernel composition (node-traced, c=64): cuDNN `Kernel2` 14%, `nchwToNhwcKernel` 12% + `nhwcToNchwKernel` 2.4% (layout transposes ≈15%), tf32 implicit-GEMM fprop ≈17%, ConvTranspose dgrad (`dgrad_engine` + sm90 dgrad) ≈12%, `_snake_kernel` 8%, transformer fmha 2.4%. The conv stack dominates; the codec transformer is small. The 10-frame conv context is 10/11 of a 1-frame ramp window's conv work and 10/35 of a full window's: a per-request conv-state cache removes it; the 72-frame transformer context matters less.
3. **Decode step 6.1 ms at 64 rows (depth).** ~1500 kernels/step (node-traced). Per step: 80 `_cp_attn_kernel` (7.8 µs at 64 rows, 10% of decode), 28 FlashInfer `BatchPrefillWithPagedKVCacheKernel` for the one-token Talker decode (18 µs each, 8%), cuBLAS nvjet small GEMMs ≈50%, fused add+RMSNorm ~190/step (8%), splitK reduces 3.5%, top-k/top-p sampling 2%. c=1: 4.17 ms/step, ~1270 kernels, inductor Triton GEMV reductions (`triton_per_fused_mm_t_view_0` …) ≈55%. (For scale: the notes' fused B32 LM+depth step on H100 is 9.18 ms; ours is 5.46 ms at c=32 on H200.)
4. **Prefill as separate steps (prefill).** 20 prefill steps/s at c=64 (~2.6 requests each), 6.4 ms each: 12.5% of wall plus ~4% idle inside them (~1,700 kernels/step node-traced; ~31 host gaps/step from the piecewise depth-loop replay and eager glue). Folding new requests' prefill rows into decode steps would remove most of it.
5. **Host (host).** The codec EOS-filter sync above (GPU-thread prepare_inputs p95 8.3 ms); otherwise iter_total ≈ step GPU time and idle between steps is small.

Placement note: all of the above runs serialized on one worker/stream; the codec (46% of GPU at c=64) next to the latency-bound decode step is the overlap candidate.
