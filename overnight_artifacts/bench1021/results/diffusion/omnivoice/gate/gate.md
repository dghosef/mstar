Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 0.87%, same-code spread (two M* PR processes) 0.00 pt, tolerance 1.00 pt; median s/word over arms 0.324

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| mstar_main | 0.87 | 128 | 0 | 0.324 | PASS | yes |
| mstar_pr | 0.87 | 128 | 0 | 0.324 | PASS | yes |
| mstar_pr_b | 0.87 | 128 | 0 | 0.324 | PASS | yes |
| ovofficial__flashinfer | 0.96 | 128 | 0 | 0.324 | PASS | yes |
| ovofficial__flashinfer_graph | 0.96 | 128 | 0 | 0.324 | PASS | yes |
| vllmomni__default | 0.91 | 128 | 0 | 0.317 | PASS | yes |
| vllmomni__fp16_mns32 | 1.05 | 128 | 0 | 0.317 | PASS | yes |
