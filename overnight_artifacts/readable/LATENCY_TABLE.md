## Before → after, per model, and where each stood when the credits ran out

Every row is the agent's own A/B on one H200, base `main` 55b0215a3 → the agent branch head, fresh server per arm, alternating order, ≥2 rounds. Latency columns are the serving-path number the model is judged on (TTFA = time to first audio, TTFT = time to first token, JCT = job completion time); throughput columns are the closed-loop rate.

**Status at cutoff** — what each agent had when the org's monthly spend limit hit at ~03:41 PDT and all eight agents were killed with HTTP 429:

- **Saturated** — the agent had profiled to a plateau and logged it, with no more cheap leads on that model.
- **In progress** — the agent was mid-lead on that model, or had just claimed it from the shared pool, when it died.
- **Blocked** — the model could not be run at all (gated weights, no HF token).
- **No change** — profiled, nothing to take.

| Model | Before (base) | After (branch head) | Δ | Status at cutoff |
|---|---|---|---|---|
| **Qwen3.5-0.8B text** (1 GPU) | c=1 679 tok/s, c=32 13.5k, c=64 15.7k | c=1 775, c=32 15.3k, c=64 17.0k (switch-interval opt-in) / 24.0k with the existing Rust frontend | c=1 +14%, c=32 +13%, c=64 +8% (or +45% on the Rust frontend) | **Saturated** — API-server-bound at c=64 (worker idles ~250 ms per wave); the only lever left is making the Rust frontend the default |
| **Qwen3.5-2B / 4B text** | — | — | — | **In progress** — claimed 02:43, no result before the cutoff |
| **Qwen3.5-9B text** (1 GPU) | c=1 208 tok/s, c=16 2.79k, c=64 8.1k | +10.6% at c=64 with a bf16 GDN state, but greedy 3/8 identical vs fp32 | measured, **not kept** | **Saturated** — GPU-bound; the only lead is a numerics change that needs a task-accuracy gate |
| **Qwen3.5-9B text, TP2** (2 GPU) | c=1 243 tok/s, c=16 2.90k, c=64 7.44k; leader step 5.4 ms | c=1 336, c=16 4.18k, c=64 11.5k; leader step 3.6 ms, now GPU-bound | +38% / +44% / +55% | **Saturated** — lockstep stress passed (8423 completed / 1660 aborted / 0 errors) |
| **Qwen3.5-0.8B image→text** | i2t c=32 5.5–6.0k tok/s, TTFT p50 181–191 ms; c=1 2048² TTFT 187 ms | i2t c=32 7.6–7.8k, TTFT p50 74–105 ms; c=1 2048² TTFT 90 ms | +34% tok/s, TTFT −50% | **Saturated** — image path is GPU-bound (ViT replay + prefill_vision) |
| **Qwen3.5-4B image→text** | i2t c=32 3.32k tok/s, TTFT p50 316–329 ms | 4.41k tok/s, TTFT p50 179 ms | +33% tok/s, TTFT −43% | **Saturated** |
| **Qwen3.5-9B image→text** | i2t c=32 2554 tok/s, TTFT p50 424 ms | 3180 tok/s, TTFT p50 271 ms | +25% tok/s, TTFT −36% | **Saturated** |
| **Omnilingual ASR 300M** | RTFx c=1 380–385, c=16 1649–1731, c=64 2006–2026 (WER 5.28–5.31%) | 519–527 / 2179–2202 / 2468–2472 (WER same) | +37% / +30% / +23% | **Saturated** — c=64 is API-process-bound (~330 req/s) |
| **Omnilingual ASR 1B** | RTFx c=1 ~178, c=16 804, c=64 1047 (WER 3.40%) | c=1 flat, c=16 ~1040, c=64 ~1980 (WER 3.43%) | +30% / +89% | **Saturated** |
| **Omnilingual ASR 3B** | RTFx c=1 127, c=16 482, c=64 593 (WER 2.37%) | 127 / 668 / 1041 (WER 2.33%) | +39% / +76% | **Saturated** |
| **Omnilingual ASR 7B** | RTFx c=1 ~110, c=16 287, c=64 308 (WER 2.64%) | ~110 / 419 / 567 (WER 2.62%) | +46% / +84% | **Saturated** on the model; one open lead (cuDNN picks a 35–40 ms legacy conv kernel for the 7B positional conv) |
| **Qwen3-Omni** (2 GPU) | t2s c=1 15.7 audio-s/s, c=8 66, c=32 ~150; TTFA c=1 0.19 s; t2t c=1 215 tok/s; audio-in TTFT c=8 0.88 s | t2s c=1 19.4, c=8 81, c=32 180–200; TTFA c=1 0.085 s; t2t c=1 240 tok/s; audio-in TTFT c=8 0.25 s | +12…+33% t2s, TTFA −55%, audio-in TTFT −72% | **In progress** — colocated 1-GPU config landed at 03:20 (+26–33%, TTFA 0.33→0.11 s), audio-in sweep still open |
| **Qwen3-TTS 0.6B** | c=64 ~130 audio-s/s; TTFA c=32 ~80 ms | c=64 ~165 audio-s/s; TTFA c=32 ~50 ms | +27% | **Saturated** — 25% of GPU-thread time is a per-window codec EOS-mask sync, but removing it only moved c=64 by ~3% |
| **Qwen3-TTS 1.7B voice clone** | c=32 88 audio-s/s; TTFA c=1 49 ms | 108 audio-s/s; TTFA c=1 29 ms (WER 0.53% both) | +23%, TTFA −41% | **Saturated** |
| **Qwen3-TTS 1.7B VoiceDesign** | — | +23% at c=64, TTFA −33% | +23% | **Saturated** |
| **Fish S2-Pro** | c=32 ~29 audio-s/s; TTFA c=1 129 ms; c=64 RTF 0.95, TTFA p95 2.0 s | c=32 ~57 audio-s/s; TTFA c=1 62 ms; c=64 RTF 0.72, TTFA p95 0.36 s | ~2× at c=32, 2.4× at c=64 | **Saturated** — AR GEMMs dominate (9.8 ms vs a 3.3 ms weight-read floor) |
| **Higgs Audio** | c=1 10.7/11.6 req/s; c=8 29.8/33.2; c=32 warm 44.5/46.0 (WER 1.50%) | c=1 14.0; c=8 44.2/43.8; c=32 final sweep 84.1/86.4 (WER 1.48%) | +22% / +37% / 1.9× at c=32 | **Saturated** — short (<4 s) clips still run the tower eagerly |
| **OmniVoice** | c=1 29.6/31.2 audio-s/s (TTFA p50 471/447 ms); c=8 109/126; c=32 129/139 (TTFA p95 11.0/10.9 s) | c=1 77.1/80.3 (TTFA 169/166 ms); c=32 149.7/145.6 (p95 7.6/7.8 s); with the split+MPS config c=32 180.7/184.9 (p95 3.3/3.4 s) | +140% at c=1, +36–40% at c=8/32 | **Saturated** — GPU-bound at c=32 (event_sync 21 of 37 ms/step) |
| **Chatterbox** | c=1 10.2 audio-s/s (TTFA 154 ms); c=32 35.4/36.7; Turbo c=32 69.4/69.6 | c=1 11.6 (TTFA 138 ms); c=32 38.0/38.5; with split+MPS c=32 57.5/61.6, Turbo 93.5/96.8 | +10% at c=1, +67% at c=32 (config) | **Saturated** — S3Gen flow solve is GPU-bound |
| **Kokoro** | c=1/c=32/c=64 235/763/897 audio-s/s | 272–309 / 830–866 / 974–976; the `kokoro_bf16.yaml` preset reaches 1252 at c=64 (WER same, corr 0.993) | +16% (opt-in preset +44%) | **Saturated** — bf16 left **off** by default: fp32 is the author's parity choice |
| **BAGEL t2i** (1 GPU) | 1024² 16.30/16.36 s; c=4 0.061/0.062 img/s | 10.16/10.22 s; c=4 0.098 img/s | −38% latency, +58% img/s | **Saturated** — GPU-bound (GEMM ~76%, paged attention 13.5%) |
| **BAGEL t2i** (2 GPU, cfg_parallel) | 12.1–12.3 s; c=4 0.087 img/s | 6.4–6.5 s; c=4 0.17 img/s | −47% latency, 2× img/s | **Saturated** |
| **BAGEL image→text** | c=1 TTFT p50 0.189 s; c=16 448 tok/s | 0.174 s; 536 tok/s | +20% tok/s | **Saturated** |
| **FLUX.2-klein-4B** | B=1 0.353/0.364 s; c=1 2.74 img/s; c=4 3.28/3.34; c=16 3.36/3.35 | 0.343/0.342 s; 2.82; 3.42/3.41; 3.42/3.31 | +2–3% | **Saturated** — GPU-bound DiT (GEMM 62%, SDPA 18%, eager LayerNorm 8%) |
| **FLUX.2-klein-9B** | — | — | — | **Blocked** — gated repo, no HF token; released 01:50 |
| **Z-Image-Turbo** | B=1 0.860/0.848 s; c=4 1.056/1.133; c=8 1.129; c=16 1.16–1.19 img/s | 0.843/0.836 s; 1.226/1.241; 1.23–1.24; 1.24 img/s | +9–12% | **Saturated** — GPU-bound DiT (GEMM 67%, SDPA 18%) |
| **Wan2.2-TI2V-5B** (1 GPU) | 20 steps 24.70 s (DiT step 1062 ms); 50 steps 56.95 s; 480p 9.33 s; mux 0.90 s | 22.43 s (step 973 ms); 51.90 s; 8.42 s; mux 0.70 s | −9% | **Saturated** — GPU 99% busy (FA3 40%, GEMM 45% at ~730 TF/s) |
| **Wan2.2** (2 GPU) | sp2 = 1-GPU; dp2 c=4 1.05 vpm (random replica 3.58) | sp2 50 steps 27.99 s (2.03×); dp2 c=4 2.31 vpm (5.21 vpm with the least-loaded pick) | 2.03× / 2.2× | **Saturated** |
| **Cosmos3-Nano** (1 GPU) | 56.1 s e2e (VAE decode 2.90 s) | 55.5 s e2e (decode 2.39 s) | −4% e2e, −18% decode | **Saturated** |
| **Cosmos3-Nano** (2 GPU) | c=2 1.07 vpm; sp2 c=1 56.1 s | dp2 c=2 2.13 vpm; sp2 c=1 31.5 s | 2× | **Saturated** |
| **Cosmos3-Edge** (1 GPU) | 6.58 s (decode 1.84 s) | 6.19–6.29 s (decode 1.51–1.54 s) | −4% e2e, −18% decode | **Saturated** |
| **Cosmos3-Edge** (2 GPU) | sp2 c=1 6.27 s (1 GPU); dp2 c=4 7.9 vpm | sp2 c=1 4.41 s (frames identical); dp2 c=4 15.8 vpm | −30% e2e, 2× vpm | **Saturated** |
| **Cosmos3-Super, TP2** | — | — | — | **In progress** — async-TP probe recovered 0.18 of 1.1 ms of comm → not worth the Megatron-SP rewrite, dropped at 04:15; no result |
| **LingBot-World 1.3B** | 9.60 fps (480×832) | 12.55 fps strict (DiT bit-exact, frames 69 dB); 18.59 fps with the opt-in fast-numerics config | +31% strict, +94% opt-in | **Saturated** |
| **V-JEPA2-AC** | c=1 JCT 0.145 s; c=4 6.5 req/s; c=8 6.9; c=16 9.7 | c=1 0.117 s; c=4 18.4; c=8 20.3; c=16 21.4 req/s | 2.8× / 2.9× / 2.2× | **Saturated** — throughput plateaus at ~20 req/s from c=8 |
| **V-JEPA2 (ViT-L)** | c=1 2.70 s (decode 2.1 s); c=8 ~0.81 req/s | c=1 2.01 s (decode 0.8 s); c=8 unchanged | −26% at c=1 | **Saturated** — c=8 is API-bound (32 MB base64 output) |
| **Waypoint 360p / 720p** | 720p c=1 23.2 ms GPU per latent (4 frames); c=2 181 fps; 360p c=8 919 fps | unchanged | none | **No change** — kernels near roofline (FA4 ~650 TF/s, GEMMs ~645 TF/s) |
| **Cosmos3 DROID policy** | — | — | — | **In progress** — claimed 02:50, no result |

## What worked

- **Capturing the graph where it was missing** — the largest single class of wins: OmniVoice backbone, Higgs audio encoder, V-JEPA2-AC rollout encoder, FLUX.2/Z-Image text encoders, Qwen3.5's rotary tables built inside the decode graph.
- **Batching across requests instead of per request** — Qwen3.5's vision encoder (one captured ViT replay for ≤8 requests), the Higgs encoder, the TTS talker decode cap 32→64 rows, codec batch buckets per window.
- **Taking host syncs and float copies off the serving path** — uint8 image preprocessing end to end, byte patches + GPU-side normalize, Qwen3-Omni thinker postprocess, OmniVoice's `.item()` and pydub trim, Chatterbox T3's per-row H2D, BAGEL's `image_gen` prepare_inputs, the DiT schedule tensors staged pinned and non-blocking.
- **Using the second GPU** — Wan2.2 CFG split (sp2) and DP2 plus the least-loaded replica pick, Cosmos3 nano/edge dp2 and sp2, BAGEL's 2-GPU parallel-CFG walk, Qwen3.5-9B TP2 with `tp_async_sched`.
- **Kernel-level work** — fused softmax-top-k MoE router, fused add+RMSNorm, block-diagonal BiLSTM, fused Triton Snake + TF32 codec, FA3 as an opaque op inside the compiled graph, GELU in the cuBLASLt epilogue, compiled channels-last VAE decode, whole-clip VAE decode.
- **Numerics kept as opt-in, not defaults** — LingBot's fast-numerics config, the `kokoro_bf16.yaml` preset, Qwen3.5's per-config API switch interval.
- **Three real bugs in `main`, found and fixed** — V-JEPA2 rollouts returning the last iteration's prediction H times (the worker's reused pinned host buffers), the opaque ragged op raising `IndexError` under compile+capture, and a cold DP replica's first request costing 10.5 s instead of 6.3 s.

## What didn't (measured, then dropped)

- **A global default that hurt one model** — the API server's 1 ms switch interval sped up Qwen3.5 but cost Kokoro −3.7%, so it became a config key set for Qwen3.5 only.
- **Capturing more of the graph** — compiling the V-JEPA2-AC rollout block-loop (c=1 0.116→0.208 s, graph breaks), capturing BAGEL's image_gen step (GEMMs already at ~720 TF/s), pushing Z-Image/klein capture buckets to 16384.
- **A kernel that wins in a microbench and loses in the server** — Qwen3-TTS's fused code-predictor kernel (−23…−32% microbench, slower served), OmniVoice's fused add+RMSNorm in the microbench.
- **Other attention backends** — FlashInfer instead of cuDNN SDPA for klein/Z-Image (klein c=4 −12%, c=16 −9%; Z-Image c=4 −6%), FA3 behind graph breaks in Wan2.2 (+0%), FA3 for Wan2.2's 512-token cross-attention (0.59 vs 0.46 ms/layer).
- **Mixed-iteration or mixed-size batches** — V-JEPA2-AC rows at different rollout iterations (c=4 18.1→11.4 req/s), Higgs's mixed walk, S3Gen at 16 rows (−10%).
- **Compile and shape work that did not pay** — whole-encoder compile for ASR (−5…−10%), `compile=True` on the captured ASR region, ASR DP sub-batching, float attention masks, Qwen's 8192/32768 token buckets, Qwen3.5 capture buckets to 16384 patches, bf16 `pixel_values`, a prefill_vision cap of 8 (+5% tok/s but +15% TTFT).
- **Padding and conv rewrites for ASR** — frontend shape padding, and frontend convs as unfold+GEMM (host 11.4→1.8 ms per new shape but only +8% served and WER 5.30→5.35%).
- **Knobs that measured as noise or worse** — `MSTAR_SAMPLER_INGRAPH_SCATTER`, `MSTAR_KV_PLAN_CACHE` (twice), the KV-admit micro-opt (34→27 µs/step, invisible end to end), the switch interval in every process (c=1 −10%) or in workers only (noise), the data-worker's multi-request preprocess pass, S3Gen host-sync removal under MPS, the compiled HiFT vocoder, the compiled flow estimator under graphs (−7% solve, +12 min startup), OmniVoice's batched CFG scoring, OmniVoice at batch 32 (−13%).
- **Numerics changes not taken by default** — Qwen3.5-9B's bf16 GDN state (+10.6% at c=64 but greedy 3/8 identical, so it needs a task-accuracy gate), LingBot on FA4 as the default (26.5 dB drift → opt-in only), Kokoro's bf16 decoder (the author chose fp32 for parity).
- **Bottlenecks that are not in the model** — Qwen3.5-0.8B at c=64 is capped by the Python API server, not the worker; the remaining lever is making the Rust frontend the default (+45%). OmniVoice, Chatterbox, Z-Image, FLUX.2, Wan2.2, Waypoint and BAGEL all ended GPU-bound, where only a cheaper kernel helps.
- **Not attempted** — FLUX.2-klein-9B (gated weights, no HF token), Qwen3.5-2B/4B text, Cosmos3-Super TP2, Cosmos3 DROID policy, and V-JEPA2 (ViT-L) beyond the shared video-decode change.
