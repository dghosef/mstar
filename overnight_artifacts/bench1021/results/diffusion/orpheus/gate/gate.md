Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.82%, same-code spread (two M* PR processes) 0.41 pt, tolerance 1.00 pt; median s/word over arms 0.372

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| mstar_main | 2.23 | 128 | 0 | 0.378 | PASS | yes |
| mstar_pr | 2.14 | 128 | 0 | 0.374 | PASS | yes |
| mstar_pr__python | 2.05 | 128 | 0 | 0.374 | PASS | yes |
| mstar_pr__rust | 2.46 | 128 | 0 | 0.370 | PASS | yes |
| mstar_pr_b | 2.55 | 128 | 0 | 0.374 | PASS | yes |
| orpheusvllm__default | 2.28 | 128 | 0 | 0.373 | PASS | yes |
| orpheusvllm__ignore_seed | 1.82 | 128 | 0 | 0.369 | PASS | yes |
| voxserve__bs64 | 2.32 | 128 | 0 | 0.347 | PASS | yes |
| voxserve__bs64_async | 2.19 | 128 | 0 | 0.341 | PASS | yes |
| voxserve__bs64_compile | 2.55 | 128 | 0 | 0.352 | PASS | yes |
