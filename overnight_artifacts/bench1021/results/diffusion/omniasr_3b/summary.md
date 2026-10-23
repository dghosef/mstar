# omniasr_3b: Omnilingual ASR CTC 3b (LibriSpeech test-clean, RTFx)



Arms: M* PR = config; M* main = config; omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) = bf16_sorted
Competitor choice: no engine serves the CTC models (vLLM PR #36864 is open and LLM-variant only; nothing in SGLang, SGLang-Omni, vLLM-Omni or transformers), so the official package's own batched pipeline runs as an offline-batch arm on the same v2 weights; batch b in the c column, length-sorted (its strongest offline setting) plus as-shipped input order as a tune knob

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## ls500: LibriSpeech test-clean (500 utterances, fixed stride), RTFx = audio s / wall s

### audio-s/s

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 242.054 [241.736-242.372] n=2 | 875.663 [863.497-887.830] n=2 | 1005.365 [1004.650-1006.079] n=2 |
| M* main | 116.831 [116.518-117.144] n=2 | 428.563 [428.469-428.657] n=2 | 573.980 [573.860-574.101] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 114.177 [113.751-114.603] n=2 | 648.362 [643.710-653.015] n=2 | 813.803 [810.728-816.877] n=2 |

### req/s

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 32.169 [32.126-32.211] n=2 | 117.761 [116.125-119.398] n=2 | 135.204 [135.108-135.300] n=2 |
| M* main | 15.527 [15.485-15.568] n=2 | 57.634 [57.622-57.647] n=2 | 77.190 [77.174-77.206] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 15.174 [15.117-15.231] n=2 | 87.193 [86.568-87.819] n=2 | 109.442 [109.029-109.856] n=2 |

### latency p50 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.032 [0.032-0.032] n=2 | 0.128 [0.126-0.130] n=2 | 0.459 [0.456-0.462] n=2 |
| M* main | 0.068 [0.068-0.068] n=2 | 0.273 [0.267-0.279] n=2 | 0.821 [0.818-0.823] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 0.067 [0.067-0.067] n=2 | 0.161 [0.157-0.165] n=2 | 0.429 [0.427-0.431] n=2 |

### latency p99 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.054 [0.054-0.054] n=2 | 0.224 [0.222-0.226] n=2 | 0.652 [0.648-0.656] n=2 |
| M* main | 0.082 [0.079-0.084] n=2 | 0.444 [0.436-0.451] n=2 | 1.042 [1.042-1.042] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 0.091 [0.088-0.094] n=2 | 0.654 [0.654-0.655] n=2 | 2.554 [2.547-2.561] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.032 [0.032-0.032] n=2 | 0.128 [0.126-0.130] n=2 | 0.459 [0.456-0.462] n=2 |
| M* main | 0.068 [0.068-0.068] n=2 | 0.273 [0.267-0.279] n=2 | 0.821 [0.818-0.823] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | - | - | - |

M* PR audio-s/s relative to each arm:

- M* PR vs M* main: c=1 +107.2%, c=16 +104.3%, c=64 +75.2%
- M* PR vs omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch): c=1 +112.0%, c=16 +35.1%, c=64 +23.5%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_main | PASS | ls500: {"wer": 2.536104262064107, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "\uc544i mi"], [0.4, 79, "tomorrow is the examination"], [0.25, 107, "money my iron's quite told please put it down to warm"]], "parity": true} |
| mstar_pr | PASS | ls500: {"wer": 2.500880591757661, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "\uc544i mi"], [0.4, 79, "tomorrow is the examination"], [0.25, 107, "money my iron's quite told please put it down to warm"]], "parity": true} |
| mstar_pr_b | PASS | ls500: {"wer": 2.4656569214512154, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "\uc544i mi"], [0.4, 79, "tomorrow is the examination"], [0.25, 107, "money my iron's quite told please put it down to warm"]], "parity": true} |
| omniasr_official__bf16_sorted | PASS | ls500: {"wer": 2.536104262064107, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "\uc544i mi"], [0.4, 79, "tomorrow is the examination"], [0.25, 107, "money my iron's quite told please put it down to warm"]], "parity": true} |
| omniasr_official__bf16_unsorted | PASS | ls500: {"wer": 2.536104262064107, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "\uc544i mi"], [0.4, 79, "tomorrow is the examination"], [0.25, 107, "money my iron's quite told please put it down to warm"]], "parity": true} |

Profile: phases_ls500_c64.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

