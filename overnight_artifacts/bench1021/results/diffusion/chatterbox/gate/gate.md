Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.14%, same-code spread (two M* PR processes) 0.32 pt, tolerance 1.00 pt; median s/word over arms 0.255

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| cbserver__default | 1.14 | 128 | 0 | 0.261 | PASS | yes |
| cbvllm__eager | 1.55 | 128 | 0 | 0.273 | PASS | yes |
| mstar_main | 1.41 | 128 | 0 | 0.253 | PASS | yes |
| mstar_pr | 1.59 | 128 | 0 | 0.255 | PASS | yes |
| mstar_pr_b | 1.91 | 128 | 0 | 0.255 | PASS | yes |
