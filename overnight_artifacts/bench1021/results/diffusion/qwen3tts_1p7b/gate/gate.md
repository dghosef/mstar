Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.37%, same-code spread (two M* PR processes) 0.46 pt, tolerance 1.00 pt; median s/word over arms 0.409

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| mstar_main | 1.82 | 128 | 0 | 0.411 | PASS | yes |
| mstar_pr | 1.37 | 128 | 0 | 0.409 | PASS | yes |
| mstar_pr_b | 1.82 | 128 | 0 | 0.414 | PASS | yes |
| sgomni__base | 1.96 | 128 | 0 | 0.400 | PASS | yes |
| sgomni__mrr128 | 1.96 | 128 | 0 | 0.400 | PASS | yes |
| vllmomni__default | 1.73 | 128 | 0 | 0.413 | PASS | yes |
| vllmomni__high_conc | 1.55 | 128 | 0 | 0.409 | PASS | yes |
