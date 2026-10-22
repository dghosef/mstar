> [!CAUTION]
> **DO NOT MERGE.** This is a tracking board for an overnight multi-agent performance push. It will never be merged. Changes that hold up will become separate, focused PRs.

**18 of 18 in-scope models worked on.** Everything is measured on H200 against `main` 55b0215a3, one A/B per change with alternating arms and fresh servers, and every change passed an output check before it was kept. Out of scope: LTX-2.5, pi0.5, Whisper, Nemotron duplex.

## Results by model

Gains are this branch vs `main`. "Measured at" lists the concurrencies actually run; everything else is untested.

| Model | What was done | Throughput | Latency (TTFT / TTFA) | Measured at | Not validated / risks before merging |
|---|---|---|---|---|---|
| **Qwen3.5-0.8B** text | RoPE tables built inside the decode graph; API-server GIL switch interval (opt-in, Qwen3.5 configs only) | **+12–16%** (c=1/32/64) | not measured | c=1, 32, 64 | c=2–16 and long prompts / chunked prefill untested. CPU-bound: the existing Rust frontend would add +45% at c=64 but isn't enabled |
| **Qwen3.5-2B / 4B** text | shared changes + a sync H2D removal | +2–5% (within noise) | not measured | c=32, 64 | gains not above noise; c=1–16 untested |
| **Qwen3.5-9B TP2 / 27B TP4** | turn on TP async scheduling (`tp_async_sched`, off on main) | 9B **+38–55%**; 27B **+22–35%** | leader step 5.4→3.6 ms | c=1, 16, 64 | changes lockstep scheduling. Stress-tested (aborts, EOS, mass cancels, 0 errors), but only for minutes; 27B ran 1 round; c=2–8 and c=32 untested |
| **Qwen3.5 image→text** (0.8B–9B) | vision encoder batched across requests (engine one-row lease); images kept uint8; byte patches normalized on the GPU | **+25–40%** at c=32; mixed text+image +25–30% | big images c=1 **−50 to −60%**; c=32 p50 −40 to −60% | c=1, 4, 32 | **outputs not byte-identical** (greedy 9/14 match; bf16 reordering), so MMMU accuracy must be checked; c=8–16 and c=64 untested |
| **Qwen3.5 video→text** | same changes ported onto the new video path (separate branch `perf/ovn-qwen35vl-video`, on newer main) | **+36% (c=32), +66% (c=8)** | **−20 to −35%** | c=1, 8, 32 | not in this PR's diff; Video-MME not run; only synthetic 8–16 s clips; long or high-res videos exceed the capture ceiling and run eagerly |
| **Omnilingual ASR** 300M–7B | feature-extractor compile; packed variable-length encoder, captured; positional conv as per-group GEMMs (cuDNN picks a slow kernel) | 300M **+23–37%**; 1B–7B **1.7–2.9×** | c=1 −27% (300M) to −66% (7B) | c=1, 16, 64 | LibriSpeech test-clean only (short English clips); long-form audio and c=2–8 / c=32 untested; the conv-as-GEMM copies its input 128× (memory at big batches) |
| **Qwen3-Omni** | host syncs removed; first audio chunk 8 frames not 25; fused MoE router + tuned tiles; code-predictor fusions; audio encoder batched | speech **+23–26%**; text +7–11%; audio→text **2.2–7.6×** | TTFA **−57%** (c=1), −52% (c=32); audio-in TTFT −40 to −60% | c=1, 8, 16, 32 | router not bit-exact (bf16 ties); shorter first chunk needs a listening check at the boundary; multi-GPU TP layouts untested (main itself hangs in one of them on a cold cache); c=64 untested |
| **Qwen3-TTS** 0.6B / 1.7B (CustomVoice, VoiceDesign, voice clone) | 64-row decode; codec batch buckets; batched stop check; fused add+RMSNorm; voice-clone prefill captured | 0.6B **+25%** (c=64); 1.7B **+11–25%** | TTFA c=32 **−30 to −35%**; clone c=1 −42% | c=1, 16, 32, 64 | greedy output isn't reproducible across processes, so quality rests on WER over 130–260 sentences; c=2–8 untested |
| **Fish S2-Pro** | prefill and decode captured; fused Snake kernel; TF32 codec (config); 64-row decode | **+73% (c=16), +95% (c=32), 2.4× (c=64)** | TTFA c=1 **−48%**; c=64 p95 −82% | c=1, 16, 32, 64 | TF32 codec changes audio slightly (~60 dB SNR vs exact): a quality call; cold start adds ~24 min of autotune; one unexplained worker stall seen once |
| **Orpheus** | audio-vocabulary-only LM head; batched stop check | +7–10% (c=16/32), **+10–20%** (c=64) | c=1 RTF −6% | c=1, 16, 32, 64 (colocated) | 2-GPU config not improved; the head is disabled under TP; c=2–8 untested |
| **Higgs Audio** | audio encoder no longer recompiles per shape, batched and captured; Qwen3.5-style decode loop | +40% (c=1), +80% (c=8), **~2.1× (c=32)** | not measured | c=1, 8, 32 | latency not reported; c=16 and c=64 untested |
| **OmniVoice** | backbone captured as CUDA graphs; host stalls removed; own layer loop with fused norms; split config (vocoder on 2nd worker + MPS) | **2.5× (c=1)**; +22% (c=8/32), **+52%** with split | TTFA c=1 **−63%**; c=32 p95 −68% (split) | c=1, 8, 32 | layer loop matches the reference to fp16 rounding, not bit for bit (>95% argmax agreement); split needs an MPS daemon; c=2–4 and c=16 untested |
| **Chatterbox** (+ Turbo, Multilingual) | per-row H2D sync removed in T3; split configs (S3Gen on a 2nd worker + MPS) | +7–18% default; **+45–58%** at c=32 with split | Turbo −12%; split **raises** TTFA at c≥8 (~400→700 ms) | c=1, 8, 32 | split trades latency for throughput and needs MPS; Multilingual WER checked on English only; the T3 fix had no separate output check |
| **Kokoro** | fused BiLSTM directions; scoped cudnn.benchmark; bf16 preset (opt-in) | **+12–20%**; bf16 preset +37% | −5% | c=1, 32, 64 | bf16 preset changes numerics (fp32 is the author's parity choice); a −5.5% seen once on the shared branch didn't reproduce (likely noise) |
| **BAGEL** | CFG branch pruning; no host syncs in prepare_inputs; single-image ViT SDPA; faster PNG encode; 2-GPU parallel-CFG config | **+60%** at c=4 (1 GPU), 2× on 2 GPUs; image understanding +20% | 1024² **−40%** (16.3→9.7 s); 6.4 s on 2 GPUs | B=1, c=4 | images differ at the cross-process noise level (PSNR ~41 dB) plus a visual check only; other resolutions and c≥8 untested |
| **FLUX.2-klein** 4B / 9B | Qwen3 text encoder captured | +1–6% | −1 to −6% | B=1, c=1–16 | small gain, GPU-bound; low risk (images bit-identical) |
| **Z-Image-Turbo** | capture every batch size 1–8, not only powers of 2 | **+12–15%** (c=4), +8% (c=8) | ~flat | B=1, c=1–16 | main itself isn't deterministic across restarts; c>16 untested |
| **Wan2.2** | FA3 inside the compiled graph (opaque op); GELU fused into the GEMM; faster mp4 mux; 2-GPU CFG split and DP2 configs | ~+9% (1 GPU); DP2 2.2× on 2 GPUs | −9% (1 GPU); **−51%** with the 2-GPU CFG split | c=1, 4 | 480p mostly untested; CFG split is 29–44 dB vs 1 GPU (the same as an attention-kernel swap) |
| **Cosmos3** (nano, edge, Super, DROID) | whole-clip VAE decode; sp2 / DP2 configs; DROID batched action step compiled; per-batch-size warmup | DROID **+30–53%** with requests in flight; DP2 2× | edge sp2 −30% e2e; VAE decode −18% | c=1, 2, 4; 1/4/8 in flight | Super only profiled (no change); batched DROID actions ~1% rel-L2 from single-request (main's eager batch is similar); warmup adds 12–60 s at startup |
| **LingBot-World** | timestep embedding broadcast; channels-last VAE decode; camera path once per chunk; fast opt-in config | **+30%** (exact); +94% with the opt-in | first chunk −18% | c=1 only | **only c=1 tested**; the opt-in drifts ~25 dB from exact |
| **V-JEPA2 / V-JEPA2-AC** | 8-thread video decode (shared); AC rollout encoder captured; RoPE tables shared | ViT-L ~flat (API-bound); AC **3–6× (c=4)**, 2× (c=8) | c=1 −10% / −19 to −31% | c=1, 4, 8, 16 | AC numbers include the worker correctness fix below; c≥32 untested |
| **Waypoint** | none kept (kernels already near roofline) | — | — | c=1 (720p), c=8 (360p) | — |

## Shared-code changes

These are merged into `perf/overnight-shared`. Each one was re-tested on Qwen3.5-0.8B, Qwen3-TTS, Z-Image and Kokoro, 4 rounds ABBA, and the CPU suite ran on a GPU box. The final combined run vs main: Qwen3.5 +12–16% (greedy identical), the others at parity.

| Change | Effect | Risk |
|---|---|---|
| Least-loaded data-parallel replica choice (conductor) | Wan2.2 DP2 +45%, Qwen3.5 DP2 +22% | low |
| Warm every DP replica / every batch size (opt-in) | first-request latency on a fresh server 1.7× | adds startup time |
| Engine "one row per step" lease for piecewise capture | enables cross-request vision batching | new engine concept |
| Opaque custom op for ragged attention (+ a fix for lazy registration) | Wan2.2 −6% per step | new op path; the fix has a regression test |
| Worker: held outputs no longer sent from a reused host buffer | **fixes data corruption** (V-JEPA2 rollouts returned one step H times) | async-worker change; ordering and Rust-path parity checked |
| Sync-free schedule staging in the DiT scaffold | +1% | low |
| 8 FFmpeg threads for request video decode | 2.5× faster decode | low |
| API-server GIL switch interval, per-config | +11% for Qwen3.5 | opt-in only (it hurt Kokoro as a global default) |
| Omni MoE router + tile sweep | Omni +5–7% | Omni-only shapes |

## Bugs found

- **Fixed:** worker data corruption for loop-accumulated outputs (pre-existing on main), and an IndexError in the new ragged op (introduced and fixed tonight). Both have regression tests.
- **Not fixed, pre-existing on main:** Qwen3-Omni with the thinker TP-split over 2 GPUs (Talker on rank 1) hangs at startup on a cold compile cache. Rank 1 compiles the Talker for 10–15 min before its first TP barrier, past the 600 s timeout.

## Risks that apply to everything

- **Tuned and measured on the same data.** There was no held-out validation, so wins under ~10% may be noise or overfit to the benchmark shape.
- **Noisy hosts.** The boxes shared physical nodes, so CPU-bound numbers carry about ±10% noise. Most model-local wins rest on 2 rounds.
- **Concurrency gaps.** c=2–16 is untested for most models.
- **H200 only.** Tile sweeps and capture sizes may differ on H100.
- **Old base.** The branches are on main 55b0215a3 and would need a rebase.
