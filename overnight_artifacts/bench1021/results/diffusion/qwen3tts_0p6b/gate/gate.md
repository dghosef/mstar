Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.82%, same-code spread (two M* PR processes) 0.14 pt, tolerance 1.00 pt; median s/word over arms 0.393

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| mstar_main | 1.82 | 128 | 0 | 0.393 | PASS | yes |
| mstar_pr | 1.96 | 128 | 0 | 0.398 | PASS | yes |
| mstar_pr_b | 1.82 | 128 | 0 | 0.393 | PASS | yes |
| sgomni__base | 2.14 | 128 | 0 | 0.390 | PASS | yes |
| sgomni__mrr128 | 2.23 | 128 | 0 | 0.387 | PASS | yes |
| vllmomni__default | 2.28 | 128 | 0 | 0.393 | PASS | yes |
| vllmomni__high_conc | 1.96 | 128 | 0 | 0.390 | PASS | yes |
