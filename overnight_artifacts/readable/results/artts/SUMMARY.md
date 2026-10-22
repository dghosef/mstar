# artts: Qwen3-TTS + Fish S2-Pro serving perf (final)

Written by the coordinator from the artts agent's final report, because the agent's tools refused summary files.

Branch `perf/ovn-artts` (/Users/josephtan/mstar-ovn-artts), rebased on shared 387ce372c, 15 commits, no rust/ changes. H200. Final A/Bs: shared head 6e9b686b5 vs this head, 2 rounds each, alternating, fresh server per arm, `benchmark.tts_speech_bench` with MSTAR_PHASE_TIMING. WER is whisper-large-v3-turbo over 130-260 sentences.

| model | metric | shared head | perf/ovn-artts |
|---|---|---|---|
| Qwen3-TTS 0.6B CustomVoice | c=1 RTF / TTFA p50 | 0.061 / 21 ms | 0.058 / 19 ms |
| | c=32 audio-s/s / TTFA | 133-136 / 75-77 ms | 139-142 / 49-51 ms |
| | c=64 audio-s/s / TTFA | 129-133 / 89-104 ms | 162-167 / 75 ms |
| Qwen3-TTS 1.7B CustomVoice | c=64 audio-s/s / TTFA c=32 | 123-127 / ~80 ms | 137-146 / 56-62 ms |
| Qwen3-TTS 1.7B VoiceDesign | c=32 / c=64 audio-s/s | 117 / 121-124 | 127-131 / 146-154 |
| Qwen3-TTS 1.7B Base (voice clone) | c=16 / c=32 audio-s/s, TTFA c=1 | 65-67 / 85-86, 50-52 ms | 83.4-83.6 / 108-110, 29 ms |
| Fish S2-Pro | c=1 RTF / TTFA p50 | 0.222 / 116-119 ms | 0.158 / 61-62 ms |
| | c=16 / c=32 / c=64 audio-s/s | 24 / 30 / 30.6 | 41.5 / 58.5 / 72.7 |

WER was unchanged throughout: Qwen3-TTS 1.4-2.1%, clone ~0.5%, Fish 0.8-1.5%. The 2 failing CPU-suite tests in test_eval_compare.py also fail on the shared base.

## Commits, oldest first
- Qwen3-TTS:
  - Talker decode at 64 rows a step.
  - Codec dense batch buckets, one window size per batch.
  - Text condition added once per decode batch.
  - Talker row outputs plus a batched stop check.
  - Every codec stream item is a [1, groups] frame. This fixes a clone-path crash the row-outputs commit had introduced.
  - Fused add+RMSNorm in the Talker and CodePredictor.
  - The clone prefill replays the packed prefill capture: c=32 +23%, TTFA 49 → 29 ms.
  - `configs/qwen3tts_split.yaml`, a throughput option: about +10% at c=32/64 with MPS, but c=1 RTF is 0.072 vs 0.060. The same split on the 1.7B gives about +17%.
- Fish S2-Pro:
  - Packed prefill capture.
  - Compiled decode capture. On an empty Inductor cache the first start autotunes for about 24 min.
  - Fused Triton Snake1d.
  - TF32 codec in the config: ~60 dB SNR against the exact codec, c=32 +35%.
  - 64-row AR decode.
  - The clone prefill replays the packed captures.
- Tests for all of the above.

## Dropped (numbers in NOTES.md)
- Qwen3-TTS: bf16 codec (×3); removing the codec's EOS host sync; a fused CodePredictor qk-norm+RoPE+KV Triton kernel (faster in a microbench, but in the server c=1 RTF went 0.058 → 0.066); `api_switch_interval_s` 1 ms.
- Fish: codec chunk_frames 64; compile_mode="default"; cudnn.benchmark; fused add+RMSNorm everywhere (c=1 RTF 0.151 → 0.181); w13 gate/up stacking (noise).

## Plateaus and leads
- **Qwen3-TTS 0.6B** is at about 165 audio-s/s at c=64, with the GPU stream ~85% busy. About 2.5 ms per step goes to per-row prepare_inputs and routing. The lead is a device-resident next-frame embedding, through a recurrent-state resource with loopback. That is engine-level work.
- **Fish** is at about 58 / 72 audio-s/s at c=32 / c=64. The decode profile at c=32: GEMMs 51%, codec convs 15%, norms 11%, attention 7%. One Inductor kernel, the Fast AR's final residual-add+RMSNorm, was autotuned to grid=(1,1,1): 116-130 µs × 10 per frame, about 1.2 ms of a 9.8 ms step. The same op at other layers takes 3 µs.
  - The lead is to put only that final norm on FlashInfer. Untested patch: harness/fish_w13_finalnorm_untested.patch.
  - Or pin that reduction's Inductor config.
  - After that, the codec ConvTranspose kernels.
- One unexplained stall: a Fish server on a guest box (host venv) accepted 53 requests while its worker sat idle in `wait_for_work` for 7+ minutes. The same build had served a full round there earlier. Not investigated.

### Orpheus (orpheus-3b-0.1-ft + SNAC 24 kHz, H200), added in the afternoon

`orpheus_colocated.yaml`, base vs final (both commits); 6 base runs and 3 final runs across 3 boxes:

| | c=1 RTF / TTFA p50 | c=16 | c=32 | c=64 audio-s/s |
|---|---|---|---|---|
| base | 0.223-0.233 / 87-93 ms | 40.5-44.4 | 58.3-61.8 | 61.3-72.1 |
| final | 0.215-0.216 / 85-86 ms | 43.5-48.1 | 61.1-66.0 | 73.4-79.0 |

- ffef7f304: custom-token-only LM head. This is the October MSTAR_ORPHEUS_AUDIO_VOCAB lead, ported with no env var.
  - It multiplies by weight[128256:] (about 18% of the head) and pads the text rows with -inf, so the sampler still sees full-width logits.
  - Always on, except under TP.
  - Tokens unchanged: in greedy rollouts of the HF checkpoint (12 sentences, 4,800 steps) the sliced head picks the same token as the full head every time.
  - On its own: c=1 RTF −6% (GPU step 2.66 → 2.49 ms), c=16 +5%, c=32 +3%.
- 3f77fd4a0: batched stop check, plus tests (the Orpheus CPU tests pass, 62/62). Stop-check time per decode step at 40-70 rows drops from 1.62 ms to 0.145 ms; c=64 71 → 79 audio-s/s.
- WER (whisper-large-v3-turbo, c=32, 260 sentences, sampled output): base 2.64-3.01% over 4 runs, final 2.93%.
- The 2-GPU `orpheus.yaml` (SNAC on its own GPU) is slower than colocated at c≥32 and isn't moved by these changes. Unprofiled guess: the SNAC stream hand-off.
- Greedy output isn't reproducible across server processes (base vs base 0/13), which is why token equality was checked offline.
- Dropped: fused add+RMSNorm in the Orpheus LM; the Fish final-norm-only patch (no gain: the slow single-program kernel doesn't reappear on a fresh compile).
- Leads (nsys at c=1, GPU-bound ~2.49 ms step):
  - FlashInfer's Llama-3.1 RoPE takes 9.7 µs per layer, against 1.8 µs for a cos/sin table: ~0.22 ms per step. The fix is a Llama-3.1-scaled table in the Position resource.
  - The down projection runs at ~2 TB/s: ~0.4 ms per step.
