# omni — Qwen3-Omni-30B-A3B (H200), overnight perf push

Box ovn-omni (2x H200), main config `configs/qwen3omni_2gpu.yaml` (Thinker + encoders GPU1, Talker + Code2Wav GPU0).
Branch `perf/ovn-omni` (worktree `/Users/josephtan/mstar-ovn-omni`), rebased on `perf/overnight-shared` 6e9b686b5.
Harness `results/omni/harness/`: `bench.py` (one server per arm, MSTAR_PHASE_TIMING=100 always), `ab.py`
(alternating arms, then Whisper-large-v3 WER of the speech vs the Thinker's own text), `asr_wer.py` (LibriSpeech
transcript WER for audio-in), `summ.py`, `phases.py`, `nsys_run.py`/`nsys_kernels.py`, `pyspy_run.py`, tuners/microbenches.
Client: benchmark.runner closed loop; `text` dataset (256-token greedy Thinker, sampled Talker), `libri` (a2s/a2t), `food101` (i2t).

## Cumulative: origin/main 55b0215a3 vs branch HEAD 25c9c1ea8 (tag fin1, 2 rounds, alternating, same box)
| workload | metric | base r0 / r1 | head r0 / r1 | change |
| --- | --- | --- | --- | --- |
| t2s c=1 | audio-s/s | 15.56 / 15.77 | 19.67 / 19.77 | +26% |
| t2s c=1 | TTFA | 0.192 / 0.186 s | 0.082 / 0.082 s | -57% |
| t2s c=8 | audio-s/s | 63.1 / 65.8 | 77.0 / 82.2 | +24% |
| t2s c=8 | TTFA | 0.357 / 0.343 s | 0.161 / 0.157 s | -55% |
| t2s c=32 | audio-s/s | 153.1 / 156.5 | 183.0 / 199.1 | +23% |
| t2s c=32 | TTFA | 0.644 / 0.623 s | 0.302 / 0.305 s | -52% |
| t2t c=1 | tok/s | 215.4 / 215.6 | 239.4 / 239.4 | +11% |
| t2t c=16 | tok/s | 1165 / 1166 | 1250 / 1241 | +7% |
| a2s c=8 | audio-s/s, TTFA | 24.7 / 25.9, 0.64 / 0.66 s | 31.2 / 31.6, 0.38 / 0.38 s | +25%, -42% |
| a2t c=16 | tok/s, TTFT | 49 / 165, 1.99 / 0.70 s | 377 / 357, 0.28 / 0.29 s | x2.2-7.6 (base recompiles the audio encoder per clip length) |
| i2t c=16 | tok/s | 1014 / 990 | 1088 / 1034 | +6% |
Colocated (`qwen3omni_colocated.yaml`, 1 GPU, tag coloc1, 2 rounds): t2s c1 11.2/12.1 -> 14.1/15.4 audio-s/s, TTFA
0.33 -> 0.11 s; t2s c8 39.8/42.6 -> 53.0/53.8; t2t c1 196 -> 222/248 tok/s; a2s c8 15.5/19.0 -> 22.6/22.7 audio-s/s.
Load avg was < 15 in fin1; c=32 numbers swing ~±10% between rounds with host load (see open issues).

Correctness (fin1): speech WER (Whisper vs Thinker text) median 0.05-0.08 base and head at c1/c8/c32 (corpus WER is
dominated by a few long/markdown replies on both arms); LibriSpeech ASR WER of the Thinker transcript 1.04% (a2s) and
0.88% (a2t) on every base and head arm; greedy text at c=1 is run-to-run identical per arm, differs base vs head after the
router change at the level of batching noise (unchanged build at c=16 diverges on 39/64 prompts).

Verdict: Talker worker GPU-bound at c<=8 before and after (event_sync ~2 ms of ~3.6 ms at c1 now); at c=32 GPU-bound at
p50 with host-load-dependent CPU stalls. Talker step: ~2300 -> ~1200 graph kernels/frame.

## Kept (commits on perf/ovn-omni, oldest first; each A/B 2 rounds alternating, fresh servers)
1. `851d718a5` Thinker decode postprocess: no boolean-mask host sync per row. + `e1a440e0c` 8-frame first Code2Wav chunk
   (ScheduledLeftContextChunkPolicy, trim by reported context; stitched audio tests) — ab1: TTFA c1 0.19->0.092 s,
   c8 0.354->0.18, c32 0.64->0.34; t2s c32 148->161 audio-s/s.
2. MoE fused softmax-topk router (shared; merged by the integrator as 8ecad036d) — ab2: t2t c1 215->231 (+7.4%), c16 +5%.
3. `e2d13b5df` code predictor fused q/k-norm+RoPE+KV write and fused embed-accumulate (bit-exact tests); `3f10d2464`
   Talker text_projection batched into the captured forward — ab2: t2s c1 16.1->18.4, c8 68->77.
4. `00c96a40b` Talker decode row_outputs + check_stop buffer; `b4301de86` fused add+RMSNorm in the code predictor;
   `eb2c45a12` swept fused-MoE tiles (shared, proposal omni-fused-moe-tiles.md) — ab3: t2s c1 +5%, c8 +9%, c32 +10%, t2t c1 +4%.
5. `540994491` Talker scratch in-graph + constant depth positions, Thinker decode row_outputs, sync-free Code2Wav codec_eos
   (GPU-thread sync was 11% of its py-spy samples) — ab4: Talker prepare_inputs 0.85->0.41 ms/step at c32, a2s c8 +8%.
6. `deb4c181e` (audio half) audio encoder eager — base recompiled it per clip length (a2t c16 TTFT 2.0 s in fin1 base_r0).
7. `5eb42f32a` cached get_device in Thinker/Talker — ab7: Talker prepare_inputs 0.43->0.25 / 0.41->0.17 ms per step.
8. `5240c9969` audio encoder batched across requests (4 clips 41.3->10.9 ms) + 2-token opening code predictor pass
   (`25c9c1ea8` cp_attention) — ab8: a2t c16 345->372 tok/s, TTFT 0.34->0.30 s; t2s neutral.

## Dropped (with numbers)
- Skinny Triton GEMM for code predictor projections: L2-warm microbench 10-65% faster than cuBLAS, in situ 4.4 us avg,
  t2s c1 -2% (ab5) -> reverted `a475510c8`.
- One-kernel small-vocab sampler (shared): 11 us vs ~25 us under nsys, but Talker gpu_exec at c1 3.67->3.63 ms only, flat
  audio-s/s (ab6) -> reverted `0f857d38c`.
- FlashInfer decode CUDA-core vs tensor-core (thinker bs1 11.6 vs 7.7 us), RoPE cos/sin table vs pos_ids (1.6 vs 1.8 us):
  nothing to take; nsys graph-node tracing had inflated both.
- Inductor max-autotune for code predictor GEMMs: equal to cuBLAS.
- (earlier idea) small-vocab, skinny GEMM: see above.

## Open issues / leads
- c=32 Talker: intermittent 24-33 ms stalls in worker register_outputs track host load (both arms of a bad round), so
  c=32 audio-s/s swings 158-205 for one build. Not a code path; GIL handoff under host CPU contention.
- Code2Wav: cuDNN conv1d layout transposes (nchw<->nhwc) ~25% of a vocoder call.
- Talker sampler seen-token staging per row (~7% of GPU-thread GIL at c32); Thinker HBM-bound at c>=16 (all experts).
- Code predictor GEMMs ~1/3 of HBM roofline: 15 depth passes x 5 layers x 4 dependent small GEMMs per frame.
- Other configs (guest work at the end of the batch, 1 round each, so treat as a check that the wins hold):
  - PD-disaggregated on 2 GPUs (`configs/qwen3omni_pd_2gpu.yaml`, untracked: prefill Thinker + encoders + Code2Wav +
    Talker on rank 0, decode Thinker on rank 1; tag pd1 on ovn-omni): t2s c1 15.85 -> 19.75 audio-s/s, TTFA 0.189 ->
    0.090 s; t2s c8 66.8 -> 82.0, TTFA 0.42 -> 0.23 s; t2t c16 1193 -> 1276 tok/s. Wins hold.
  - thinker_tp2 / full_tp2 / pd_disaggregated need 3-4 GPUs. A 2-GPU thinker-TP2 variant (Talker + Code2Wav on TP
    rank 1) hangs at startup on main 55b0215a3 as well as on the branch, whenever the Inductor cache is cold. Rank 0
    waits in the Thinker's per-spec TP barrier (size_captures) while rank 1 spends >600 s autotuning the Talker. With a
    warm cache, main passed. Pre-existing; details in results/omni/thinker_tp2_hang.md.
