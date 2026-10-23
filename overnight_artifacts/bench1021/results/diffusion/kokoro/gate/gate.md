Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.28%, same-code spread (two M* PR processes) 0.14 pt, tolerance 1.00 pt; median s/word over arms 0.329

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| kokorofastapi__workers1 | 1.28 | 128 | 0 | 0.310 | PASS | yes |
| kokorofastapi__workers4 | 1.28 | 128 | 0 | 0.310 | PASS | yes |
| mstar_main | 2.32 | 128 | 0 | 0.329 | PASS | no (annotated) |
| mstar_pr | 2.32 | 128 | 0 | 0.329 | PASS | no (annotated) |
| mstar_pr_b | 2.46 | 128 | 0 | 0.329 | PASS | no (annotated) |
