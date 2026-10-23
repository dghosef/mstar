**How regressions were prevented.** Every change went through the same gates:
1. **Output check against the base arm** before it was kept: identical greedy tokens, WER over many samples (TTS/ASR), PSNR/SSIM at a fixed seed against the cross-process noise floor (diffusion, video), or bit/byte-identical outputs. A row marked *A/B only* had no separate output check recorded; treat it as unverified for quality.
2. **A/B on the agent's own box** against the base, with alternating arms and fresh servers. Two or more rounds for anything claimed, and host load recorded for CPU-bound runs.
3. **Scope.** Model-local code or a config only affects that model. A behaviour that helped one model and hurt another became a **per-config opt-in** instead of a default.
4. **Shared code** (engine, worker, conductor, API server, components) was merged into `perf/overnight-shared` only by the integrator. The integrator reviewed it against AGENTS.md, ran the CPU test suite on a GPU box (failures compared against base), and ran a cross-model regression suite (4-round ABBA): Qwen3.5-0.8B text c=1/32/64 with greedy-identity, Qwen3-TTS c=16 with WER, Z-Image B=1 with PSNR, and Kokoro c=32 with WER. Worker changes also had async-worker ordering and Python/Rust-path parity checked.
5. **Rebase as regression test.** After each shared merge every agent rebased and re-measured its own model on the new shared head, so each shared change was also checked on every model.

Status: **shared** = merged into `perf/overnight-shared`; **opt-in** = off unless a config turns it on; otherwise model-local on the agent's branch.

### Qwen3.5 (text, vision) and Omnilingual ASR
| Model | What the optimization is | Result | How regressions were prevented | Status |
|---|---|---|---|---|
| Qwen3.5-0.8B text | Build the RoPE tables inside the decode CUDA graph instead of eagerly on the GPU thread's critical path (existing knob, now the default) | c=32 +13%, c=1 +14%, c=64 flat | Greedy output identical; model-local (Qwen3.5 configs only) | model |
| Qwen3.5-0.8B text | Lower the API-server process's GIL switch interval to 1 ms, so the event loop, message and data threads hand off faster | c=64 +10.9% (4/4 rounds), c=32 +3.5% | Proposed as a global default and **rejected**: Kokoro c=32 −3.7% in two 4-round runs. Merged as a per-config key (`api_switch_interval_s`) set only in Qwen3.5 configs; other models were re-measured unchanged | **shared**, opt-in |
| Qwen3.5-0.8B text | Measured the existing opt-in Rust HTTP frontend (`--rust-frontend`); no code change | c=64 +32% | n/a; deployment option only | not changed |
| Qwen3.5-9B text, TP2 | Turn on TP async scheduling (`tp_async_sched`, off by default on main): the leader speculates the next step for the lockstep pair | +38% (c=1), +44% (c=16), +55% (c=64) | Greedy 8/8 identical at c=1, c=8 same as base; lockstep stress test (aborts, EOS, long prompts) was **not finished** when the agents stopped, so treat it as unvalidated | config, pending |
| Qwen3.5 image→text (0.8B, 9B) | Batch the vision encoder across requests into one captured step (engine opt-in: a piecewise region can lease the whole step as one row) | 0.8B c=32 +24%; 9B c=32 +25%, TTFT p50 −36% | Integrator suite: Qwen3.5 text greedy 8/8, Qwen3-TTS c=16 +1% (WER same), no new CPU test failures; text→text re-measured unchanged on 9B. **Correction:** image→text greedy output is not byte-identical (9/14 prompts; bf16 reordering from the in-node ViT compile); task-accuracy (MMMU / Video-MME) check owed | **shared** |
| Qwen3.5 image→text | Keep request images uint8 end to end and stop shipping the raw float image to workers over SHM | preprocess 68→39 ms (2048²), c=32 +~8% | pixel_values bit-identical vs base | model |
| Qwen3.5 image→text | Ship byte patches and normalize on the GPU | c=1 TTFT on big images −50 to −60% | Pixels bit-identical; confirmed over a 3rd round | model |
| Omnilingual ASR 300M | torch.compile only the conv feature extractor (its fp32 channel LayerNorm was ~45% of a step) | c=64 +23%, c=16 +13% | WER same; c=1 unchanged | model |
| Omnilingual ASR 300M–7B | Packed varlen encoder through the ragged attention resource, captured per frame bucket | 300M c=1 +37%; 1B c=64 +89%; 3B/7B c=64 +76%/+84% | WER unchanged per size (e.g. 1B 3.40→3.43%, 7B 2.64→2.62%) | model |

### Qwen3-Omni (2×H200 unless noted)
| What the optimization is | Result | How regressions were prevented | Status |
|---|---|---|---|
| Removed a per-row host sync (boolean-mask indexing) in thinker decode postprocess; first Code2Wav chunk 8 frames instead of 25 | TTFA c=1 0.19→0.09 s, c=32 0.64→0.34 s; t2s c=32 +8% | Audio WER medians unchanged | model |
| Fused softmax + top-k MoE router kernel instead of torch.topk | t2t c=1 +7.4%, c=16 +5% | Router GPU tests 16/16 against torch; only Omni imports the MoE component; CPU tests clean. Not bit-exact (bf16 ties order differently), so WER checked | **shared** |
| Code predictor: fused qk-norm + RoPE + KV write, fused embed + accumulate, text projection moved inside the captured forward | t2s c=1/c=8 +14% | Audio WER unchanged | model |
| Talker returns outputs as row outputs + one check_stop buffer (was 3 clones + 1 D2H per row per step); fused add+RMSNorm; fused-MoE tiles tuned for Omni's two shapes | t2s c=8 +9%, c=32 +10% | Audio WER unchanged; ASR WER of the audio-in path 1.04% on base and every arm | model |
| Audio encoder batched across requests (packed clips, attention windowed per clip); code predictor opens with one 2-token causal pass | audio→text c=16 +8%, TTFT 0.34→0.30 s; speech out within noise | ASR WER of the transcripts checked | model |
| Same changes on the 1-GPU colocated config | t2s +26–33%, TTFA 0.33→0.11 s | WER medians unchanged | model |

### TTS
| Model | What the optimization is | Result | How regressions were prevented | Status |
|---|---|---|---|---|
| Qwen3-TTS 0.6B/1.7B | Raise the Talker decode cap 32→64 rows; codec batch buckets per window size | c=64 +20–36%; TTFA c=32 −31 to −39% | WER within the same band over many samples (greedy TTS isn't reproducible across processes, so no token identity) | model |
| Qwen3-TTS 0.6B | Talker row outputs + batched check_stop | c=32 +4%, c=64 +3% | 3 rounds at host load < 45; later caught and fixed a voice-clone crash it introduced | model |
| Qwen3-TTS 1.7B voice clone / VoiceDesign | Clone prefill replays the packed prefill capture; fused add+RMSNorm in the Talker and code predictor | clone c=32 +23%, TTFA c=1 49→29 ms; VoiceDesign c=64 +23% | WER 0.53→0.53% | model |
| Fish S2-Pro | Captured prefill, compiled decode capture, fused Triton Snake, TF32 codec (config), 64-row decode | c=32 ~2×, c=64 2.4×; TTFA c=1 129→62 ms | WER ~1% throughout; greedy output reproducible per build (20/20); codec ~60 dB SNR vs exact fp32 | model + config |
| Higgs Audio | Audio encoder: no per-shape recompiles on the serving path, cross-request batching, piecewise capture; short clips padded onto a second captured graph | c=32 ~1.9×, c=1 +46% | WER 1.50→1.48%; short-clip padding exact up to kernels | model |
| OmniVoice | Backbone captured as CUDA graphs; two worker-loop host stalls removed (`.item()`, pydub silence trim inside the forward) | c=1 TTFA 452→180 ms; c=32 +14%, c=8 +25% | Fixed-seed waveforms bit-identical (156/156); WER 0.86→0.64%; trim sample-exact vs pydub | model |
| OmniVoice | Vocoder on its own worker with MPS (new config) | c=32 →~189 audio-s/s, TTFA p95 9→3 s | New config only, default unchanged; c=1 −10% noted as the trade-off | config |
| Chatterbox | Removed a per-step pageable H2D copy (a stream sync) in T3's position lookup | c=1 +10%, c=32 +4% | A/B only | model |
| Chatterbox | S3Gen on a second worker with MPS (config) | c=32 +60–67% | WER 1.22→1.08%; new config only | config |
| Kokoro | Fused BiLSTM directions; scoped cudnn.benchmark for the frame half | c=64 +16% | Text half bit-identical; WER same | model |
| Kokoro | `kokoro_bf16.yaml` preset (existing bf16 decoder option) | +26% over the fp32 head | WER same, waveform corr 0.993; fp32 default unchanged (product call) | opt-in preset |

### Image and video generation, world models
| Model | What the optimization is | Result | How regressions were prevented | Status |
|---|---|---|---|---|
| BAGEL | CFG branch pruning: skip guidance branches that don't change the result | 1024² 16.3→10.2 s (−37%); c=4 +60% | PSNR 41 dB median vs base, the same as base vs itself in another process (40 dB); SSIM ≥ 0.989 | model |
| BAGEL (2 GPUs) | Pruning applied to the parallel-CFG walk | 1024² 6.4 s; c=4 2× | 40 dB vs 1-GPU base (cross-process floor) | config |
| BAGEL | Cached per-device/shape index tensors in image_gen `prepare_inputs` (a host sync every step stopped speculation) | −1.2% | Bit-identical | model |
| BAGEL image understanding | Plain SDPA for a single image (was a block-diagonal masked kernel) | c=16 +20% | ViT features vs an fp32 run: same error as before (3.4–4.1e-2); greedy text drifts at bf16 level, judged quality-neutral | model |
| FLUX.2-klein | Qwen3 text encoder captured as a CUDA graph | c=1 +6% | Images bit-identical | model |
| FLUX.2-klein / Z-Image | DiT scaffold stages each request's schedule tensors pinned + non_blocking (was a stream sync) | +1–1.6% | Bit-identical; integrator: Z-Image PSNR within the floor, CPU tests clean | **shared** |
| Z-Image-Turbo | Capture every batch size 1–8 instead of powers of 2, so 3/5–7-row steps aren't padded | c=4 +15%, c=8 +8% | Same VRAM; A/B only for outputs (padding rows don't change real rows) | config |
| Wan2.2 | FA3 attention behind a torch.library custom op, so the compiled DiT has no graph break per layer | DiT step −6% | PSNR within the spread of a plain attention-kernel swap; ragged GPU tests 81/81; integrator: Qwen3.5 c=64 +0.8%, Z-Image +0.2% | **shared** |
| Wan2.2 | GELU fused into the up-proj GEMM epilogue; mp4 mux from a contiguous tensor | DiT step −1.6%; response mux −20 to −45% | Identical output bytes | model |
| Wan2.2 | CFG split across 2 GPUs (each rank one guidance branch) | c=1 −42 to −51% | PSNR 29–44 dB vs 1 GPU (the spread of a kernel swap); new config | config |
| Wan2.2 / any DP deployment | Conductor sends a new request to the least-loaded data-parallel replica instead of a random one; warmup hits every replica | Wan2.2 DP2 c=4 +45%; Qwen3.5 DP2 c=32 +22% | Integrator: replicas balanced, CPU tests clean, single-replica configs unchanged | **shared** |
| Cosmos3 nano/edge | Whole-clip VAE decode on the shared channels-last decoder; DP2 config; edge sp2 | decode −18%; DP2 2×; edge sp2 −30% e2e | Edge sp2 frames bit-identical to 1 GPU; the nano sp2 drift was checked and found to be kernel noise amplified by guidance 6, not a bug | model + config |
| Cosmos3 DROID policy | Concurrent action requests batch into a denoise step that ran eagerly while the single-request step was compiled; now compiled too | edge +44–50%, nano +25% with 4–8 in flight | Single-request path bit-identical | model |
| LingBot-World | Broadcast the scalar-timestep modulation instead of per-token copies; compiled channels-last VAE decode; camera path once per chunk | +31% fps | DiT bit-exact; frames 69 dB vs base | model |
| LingBot-World | Opt-in fast config: FA4 attention + compiled pointwise ops | +48% more | Default path unchanged and bit-exact; the fast path drifts ~25 dB, documented as the opt-in's trade-off | opt-in config |
| V-JEPA2-AC | Rollout encoder captured as batched CUDA graphs; RoPE tables shared across the 24 predictor blocks | c=4 2.8×, c=8 2.9× | Byte-identical vs eager (compile_mode default kept, since max-autotune moved outputs ~4%) | model |
| V-JEPA2 (+ Qwen3.5/Omni video) | Decode request video with 8 FFmpeg threads in the shared `Model.load_video` | c=1 2.70→2.01 s | Frames identical; integrator: one preprocess thread, so no oversubscription; CPU tests clean | **shared** |

**Measured, no gain:** TP async scheduling on the video lockstep configs (Cosmos3 edge/nano sp2, Cosmos3-Super TP2, Wan2.2 sp2): DiT step within ±0.5%, frames bit-identical, lockstep stress with cancels passed. Each step is 0.12–1.3 s of GPU work, so the serial protocol it removes is under 1%. Left off for diffusion.

### Pre-existing bug found in main (not fixed)
- **Qwen3-Omni with the thinker on 2 GPUs (TP2) and the Talker on TP rank 1 hangs at startup with a cold Inductor cache, on main 55b0215a3 too.** Each worker warms its own nodes in order, and every Thinker capture spec starts with a TP barrier. Rank 0's first node is the Thinker, so it enters the barrier at once. Rank 1 warms Code2Wav and the Talker first, and a cold Talker compile (10–15 min) outlasts the 600 s collective timeout. Both ranks run the same collectives in the same order; the problem is unbounded non-collective work before rank 1's first one (AGENTS.md invariant 11, the "when" part). It passes with a warm cache. The shipped 3-GPU `qwen3omni_thinker_tp2.yaml` keeps the Talker off the TP ranks, so it isn't affected. Possible fixes: warm the TP nodes first on every rank, or give warmup barriers a startup-length timeout. Write-up: results/omni/thinker_tp2_hang.md.

### Bugs found and fixed
- **Worker (pre-existing on main):** a loop's accumulated outputs (V-JEPA2 rollouts) were sent from the stop check's reused pinned host buffers, so a rollout returned one iteration's prediction H times. Fix: drop the step's host copies once its sends have run. Guarded by a test that fails without the fix; the integrator also checked async ordering, Rust-arena parity (tests pass with Rust on and off) and overhead (Qwen3.5 c=64 −0.5%, Qwen3-TTS/Omni neutral). **shared**
- **Opaque ragged op (introduced tonight, found by another agent):** IndexError when a node builds its callable lazily inside a compiled region. Fix: register eagerly. Guarded by a test that fails without the fix; ragged suite 56/56. **shared**

### Combined shared-branch regression (final: shared head a51da9261 vs main, 4 rounds ABBA, host load recorded)
Qwen3.5-0.8B text c=64 **+12.3%** (every head round above every main round), c=32 +11.9%, c=1 +15.7%, greedy 8/8 identical. Z-Image B=1 −0.2%, c=4 +1.3%, PSNR within the floor. Qwen3-TTS c=16 parity. Kokoro c=32 −5.5% in the integrator's run (WER identical), bisected on a quiet host to the worker held-output fix (−6.9%). **A follow-up didn't reproduce it:** revert vs head +1.9% and no-op vs head +1.3%, 4 rounds ABBA each, both inside the ±4–6% spread. The suspected mechanism was also ruled out: an instrumented c=32 round sent 100/100 outputs from the host copy, with 0 device fallbacks, and no phase grew. Verdict: unresolved, most likely host noise; the fix stays. CPU tests: head and main fail exactly the same 8 timing-sensitive tests.
