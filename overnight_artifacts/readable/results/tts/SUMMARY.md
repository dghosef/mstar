# tts: Higgs Audio, OmniVoice, Chatterbox (+ turbo, multilingual), Kokoro (final)

Written by the coordinator from the tts agent's final report, because the agent's tools refused summary files. Branch `perf/ovn-tts` (/Users/josephtan/mstar-ovn-tts), on perf/overnight-shared 6e9b686b5, 19 commits, nothing pushed. All model code or configs: no rust/, engine or worker changes. H200; base = 6e9b686b5; "split" = the `*_split.yaml` config + CUDA MPS.

| model | metric | base | head, default config | head + split |
|---|---|---|---|---|
| Higgs STT | c=32 req/s (200 req, warm) | 41.8 / 48.6 | 95.0 / 97.2 | – |
| | c=8 / c=1 req/s | ~30 / ~11 | 53.5 / 15.4 | – |
| OmniVoice | c=1 audio-s/s (TTFA p50) | 32.6 (434 ms) | 81.4 (160 ms) | 81.9 / 79.6 (165 ms) |
| | c=8 | 128 | 156 | 194.9 / 192.9 |
| | c=32 (TTFA p95) | 134.8 (9.0 s) | 163.8 | 206.8 / 205.0 (2.9 s) |
| Chatterbox | c=1 / c=8 / c=32 | 10.8 / 27.7 / 40.3 | 11.9 / 29.8 / 43.1 | 12.5 / 41 / 63.5 |
| Chatterbox Turbo | c=1 (TTFA p50) | 14.6 / 14.7 (66 / 62 ms) | 17.2 (58 ms) | 17.3 / 16.9 (62 ms) |
| | c=8 / c=32 | 51–52 / 82.5, 78.1 | 57.8 / 93.6 | 76.5–80.1 / 124.4, 113.0 |
| Chatterbox Multilingual | c=1 (TTFA p50) | 10.8 / 10.5 (100 ms) | 11.6 (95 ms) | 12.4 / 13.0 (118–120 ms) |
| | c=8 / c=32 | 28 / 39.5, 39.4 | 29.6 / 41.0 | 38.2–38.7 / 60.3, 61.2 |
| Kokoro | c=1 / c=32 / c=64 | 268 / 797 / 918 | 322 / 894 / 1037 | bf16 preset: c=64 ~1260 |

- The OmniVoice "head" column is from before the layer-loop commit; the split column includes it.
- The split configs trade TTFA for throughput at c≥8: Chatterbox c=8 TTFA p50 goes from ~400 to ~700 ms, Turbo c=32 from 0.29 to 0.8–1.0 s.

## Kept
- **Higgs:**
  - The decode loop adopts Qwen3.5's hooks (token stays on device, inline client copy, batched stop check): c=32 +12%, c=1 transcripts identical (2ba9bd5de).
  - The audio encoder is taken out of torch.compile (which recompiled for every new chunk count while serving), batched across requests and captured, including short clips (185288396, 1172dc83a).
  - Text and audio prefill as one span (4e9467868).
- **OmniVoice:**
  - Layer-stack CUDA-graph capture; host stalls removed; a sample-exact numpy silence trim instead of pydub; split config (819999a7c, 515db5663, 0d1b6f329, cbb9d7bb4, 8279fb4c7, cb7bcfee1).
  - Its own layer loop over the fused weights, with q/k norms read off the fused QKV and fused add+RMSNorm (56ecfe3f6): +5–8% at c=1–32.
  - Trade-off: logits now match the reference to fp16 rounding, not bit for bit. The parity test checks closeness plus over 95% argmax agreement. WER 0.85% → 0.93%, within each arm's 0.3 pp spread.
- **Chatterbox (all three variants):** the T3 per-row H2D fix (e551a376a) and the split configs (b8d40a981).
  - Turbo c=32 +45% with split; Multilingual +54%.
  - WER within noise: Turbo 0.93/1.08/0.88%, Multilingual 1.31/1.49/1.33%. Multilingual was checked with English prompts only.
- **Kokoro:** fused BiLSTM directions (bit-identical), scoped cuDNN benchmarking for the vocoder half, and a bf16 preset config (7767be4e4, 921fb5c79, e820b90f4).
- Docs: notes in docs/models.rst.

## Dropped
- Higgs combined prefill+decode walk: no gain (results/tts/higgs_mixed_walk_dropped.patch).
- Chatterbox T3 batched embedding + batched stop check: +1–3%, within noise (results/tts/chatterbox_t3_rows_dropped.patch).
- Earlier S3Gen, OmniVoice and Kokoro attempts: listed in NOTES.md.

## Tests
- GPU: test/omnivoice 63 passed on CUDA, including the encode/head parity test; targeted Higgs and Chatterbox modules passed.
- CPU: 596 passed on the final head, 531 on base, no failures on either (chatterbox, kokoro, omnivoice and related test/modular modules).
- The full test/modular run didn't complete on either tree: an xdist collection-order mismatch in test_piecewise_config_signature.py, and a serial hang at ~21% on both base and head. This predates the branch.

## Leads
- Chatterbox: the S3Gen flow solve bounds c=32, and ~70% of each streamed chunk's solve re-runs the 314-frame reference prompt. Changing the chunk schedule or giving first chunks priority in S3Gen batching are product or scheduler calls.
- OmniVoice: GPU-bound now. The shared ragged-attention resource copies a strided v and clones its output in graph mode on every layer. An out= or strided-input path there would remove two copies per layer for every packed model (engine proposal).
- Higgs: decode ~3 ms per step at batch 14; the remaining cost is the engine's per-step work.
- Kokoro: GPU-bound. In fp32, cuDNN layout transposes are ~19% of the vocoder half's GPU time; the bf16 preset avoids most of that.
