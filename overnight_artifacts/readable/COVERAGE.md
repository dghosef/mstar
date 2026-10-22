What was measured and checked, per model. All on H200 (Eras private slices, shared physical nodes), against `main` 55b0215a3 or the shared head at the time. **Not covered** lists the gaps: a win at the measured points can hide a loss at an unmeasured one.

| Model | Workloads | Concurrency measured | Rounds | Output check | Tests run (on a GPU box) | Not covered |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B text | text→text, 256 out | c=1, 32, 64 | 2–4 (4 in the integrator suite) | greedy identical | CPU suite on each shared merge | c=2–16; long prompts / chunked prefill |
| Qwen3.5-2B / 4B text | text→text | c=32, 64 | 2–3 | greedy | — | c=1–16 |
| Qwen3.5-9B / 27B text, TP2 / TP4 | text→text | c=1, 16, 64 | 2 (9B), 1 (27B) | greedy 8/8 vs serial at c=1, c=8 | lockstep stress: 48-lane mixed load with aborts, EOS on/off, long prompts; mass cancels (6×64 on 9B, 4×64 on 27B); 0 errors | c=2–8, c=32; 27B only 1 round |
| Qwen3.5 image→text (0.8B–9B) | image→text, mixed text+image | c=1 (TTFT by image size), c=4, c=32 | 2–3 (9B 1) | pixels bitwise; greedy text NOT byte-identical (9/14), bf16 reordering | CPU test for byte-patch normalize; 90 Qwen3.5 CPU tests + 151 engine tests on new main | c=8–16, c=64; MMMU / Video-MME accuracy not run. video→text: c=1, 8, 32 on new main (branch perf/ovn-qwen35vl-video, separate from this PR) |
| Omnilingual ASR 300M–7B | LibriSpeech test-clean 500 | c=1, 16, 64 | 2, clean sequential rounds for 3B/7B | WER per size | — | c=2–8, c=32; long-form audio |
| Qwen3-Omni | text→speech, text→text, audio→speech, audio→text, image→text; 2-GPU, colocated 1-GPU, PD-split 2-GPU | c=1, 8, 16, 32 | 2 (PD split 1) | speech WER (Whisper), ASR WER 1.04% / 0.88%, greedy text per arm | router GPU tests 16/16; fused-MoE GPU tests | TP layouts: the 2-GPU thinker-TP2 variant hangs at startup with a cold cache on main too (pre-existing; see the bug section); c=2–4, c=64 |
| Qwen3-TTS 0.6B / 1.7B (CustomVoice, voice clone, VoiceDesign) | text→speech | c=1, 16, 32, 64 | 2–3 | WER over many samples (greedy isn't reproducible across processes) | — | c=2–8 |
| Fish S2-Pro | text→speech | c=1, 16, 32, 64 | 2 | WER ~1%; greedy reproducible per build (20/20); codec SNR vs exact fp32 | — | c=2–8; cold-start cost of the compiled decode (+6 min autotune) |
| Higgs Audio | audio→text (LibriSpeech) | c=1, 8, 32 | 2 | WER | — | c=2–4, c=16, c=64 |
| OmniVoice | text→speech | c=1, 8, 32 | 2 | WER; fixed-seed waveforms bit-identical (156/156) | parity + silence-trim tests | c=2–4, c=16, c=64 |
| Chatterbox | text→speech | c=1, 8, 32 | 1–2 | WER for the split config; T3 change A/B only | — | turbo / multilingual variants; c=2–4, c=16, c=64 |
| Kokoro | text→speech | c=1, 32, 64 | 2–4 (4 in the integrator suite) | WER; text half bit-identical | — | c=2–16 |
| BAGEL | text→image 1024², edit, image understanding; 1 and 2 GPUs | B=1, c=4 (understanding c=16) | 2 | PSNR/SSIM vs the cross-process floor; edits checked by eye | — | c=2, c=8+; other resolutions |
| FLUX.2-klein 4B | text→image | B=1, c=1, 4, 8, 16 | 2–3 | images bit-identical | — | 9B (gated weights) |
| Z-Image-Turbo | text→image | B=1, c=1, 4, 8, 16 | 2–4 (4 in the integrator suite) | PSNR vs the cross-process floor | integrator suite | c>16 |
| Wan2.2 | 1280×704×81, 20 and 50 steps; 1 GPU, CFG split on 2, DP2 | c=1, c=4 | 2 | PSNR vs 1 GPU / vs base; mux output bytes identical | ragged GPU tests 81/81 (opaque op) | c=2, c=8; 480p only partly |
| Cosmos3 nano / edge / Super | 832×480 video; 1 GPU, sp2, DP2, Super TP2 | c=1, 2, 4 | 1–2 | PSNR; edge sp2 frames bit-identical | — | c=8+; Super only profiled |
| Cosmos3 DROID policy | action chunks | 1, 4, 8 in flight | 2 | actions bit-identical at 1 in flight; batched rel-L2 vs base's | warmup tests | — |
| LingBot-World 1.3B | 480×832, 93 frames | c=1 | 2 | DiT bit-exact, frames 69 dB | — | c>1 |
| V-JEPA2 / V-JEPA2-AC | embedding / AC rollout | c=1, 4, 8, 16 | 2 | byte-identical at c=1 | worker-fix test (fails without the fix); shm/tensor-comm tests with Rust on and off | c=32+ |
| Waypoint | 720p and 360p interactive | 720p c=1; 360p c=8 | 2 | n/a (no code change kept) | — | other concurrencies |

**Shared-code merges:** each was checked by the integrator's cross-model suite. That suite ran 4-round ABBA with fresh servers on Qwen3.5-0.8B c=1/32/64, Qwen3-TTS c=16, Z-Image B=1 and Kokoro c=32, plus the CPU test suite on a GPU box (failures compared against main). Worker changes also had async-ordering and Python/Rust-path checks.

**Profiles:** every model had at least one `MSTAR_PHASE_TIMING` profile pass (CPU- vs GPU-bound verdict). GPU-bound models also had nsys or torch-profiler kernel breakdowns.
