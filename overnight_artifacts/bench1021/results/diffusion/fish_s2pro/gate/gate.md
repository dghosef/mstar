Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.14%, same-code spread (two M* PR processes) 0.05 pt, tolerance 1.00 pt; median s/word over arms 0.357

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| mstar_pr | 1.64 | 128 | 0 | 0.361 | PASS | yes |
| mstar_pr_b | 1.59 | 128 | 0 | 0.358 | PASS | yes |
| sgomni__base | 1.50 | 128 | 0 | 0.358 | PASS | yes |
| sgomni__mrr128 | 1.64 | 128 | 0 | 0.355 | PASS | yes |
| vllmomni__default | 1.14 | 128 | 0 | 0.339 | PASS | yes |
| vllmomni__high_conc | 1.23 | 128 | 0 | 0.336 | PASS | yes |
