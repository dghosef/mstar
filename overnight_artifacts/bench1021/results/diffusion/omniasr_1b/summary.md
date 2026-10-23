# omniasr_1b: Omnilingual ASR CTC 1b (LibriSpeech test-clean, RTFx)



Arms: M* PR = config; M* main = config; omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) = bf16_sorted
Competitor choice: no engine serves the CTC models (vLLM PR #36864 is open and LLM-variant only; nothing in SGLang, SGLang-Omni, vLLM-Omni or transformers), so the official package's own batched pipeline runs as an offline-batch arm on the same v2 weights; batch b in the c column, length-sorted (its strongest offline setting) plus as-shipped input order as a tune knob

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## ls500: LibriSpeech test-clean (500 utterances, fixed stride), RTFx = audio s / wall s

### audio-s/s

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 268.201 [267.397-269.006] n=2 | 1526.160 [1500.468-1551.852] n=2 | 1788.621 [1771.235-1806.008] n=2 |
| M* main | 155.373 [155.266-155.479] n=2 | 739.806 [730.290-749.321] n=2 | 1031.413 [1029.626-1033.200] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 154.489 [154.402-154.576] n=2 | 941.491 [934.067-948.915] n=2 | 1208.301 [1208.158-1208.444] n=2 |

### req/s

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 35.643 [35.537-35.750] n=2 | 205.242 [201.787-208.697] n=2 | 240.538 [238.200-242.876] n=2 |
| M* main | 20.649 [20.635-20.663] n=2 | 99.491 [98.211-100.771] n=2 | 138.707 [138.467-138.947] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 20.531 [20.520-20.543] n=2 | 126.614 [125.616-127.612] n=2 | 162.495 [162.476-162.515] n=2 |

### latency p50 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.027 [0.027-0.027] n=2 | 0.075 [0.074-0.076] n=2 | 0.263 [0.260-0.267] n=2 |
| M* main | 0.048 [0.048-0.048] n=2 | 0.155 [0.153-0.158] n=2 | 0.451 [0.451-0.451] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 0.048 [0.048-0.048] n=2 | 0.113 [0.110-0.116] n=2 | 0.286 [0.285-0.287] n=2 |

### latency p99 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.044 [0.044-0.044] n=2 | 0.134 [0.129-0.139] n=2 | 0.345 [0.342-0.347] n=2 |
| M* main | 0.064 [0.064-0.064] n=2 | 0.257 [0.253-0.260] n=2 | 0.611 [0.601-0.621] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 0.067 [0.067-0.067] n=2 | 0.454 [0.439-0.469] n=2 | 1.664 [1.664-1.665] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.027 [0.027-0.027] n=2 | 0.075 [0.074-0.076] n=2 | 0.263 [0.260-0.267] n=2 |
| M* main | 0.048 [0.048-0.048] n=2 | 0.155 [0.153-0.157] n=2 | 0.451 [0.451-0.451] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | - | - | - |

M* PR audio-s/s relative to each arm:

- M* PR vs M* main: c=1 +72.6%, c=16 +106.3%, c=64 +73.4%
- M* PR vs omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch): c=1 +73.6%, c=16 +62.1%, c=64 +48.0%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_main | PASS | ls500: {"wer": 3.3110250088059177, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "y me"], [0.4, 89, "ben looking up tombs county"]], "parity": true} |
| mstar_pr | PASS | ls500: {"wer": 3.240577668193026, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "y me"], [0.4, 89, "ben looking up tombs county"]], "parity": true} |
| mstar_pr_b | PASS | ls500: {"wer": 3.3110250088059177, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "y me"], [0.4, 89, "ben looking up tombs county"]], "parity": true} |
| omniasr_official__bf16_sorted | PASS | ls500: {"wer": 3.240577668193026, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "y me"], [0.4, 89, "ben looking up tombs county"]], "parity": true} |
| omniasr_official__bf16_unsorted | PASS | ls500: {"wer": 3.2758013384994715, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "y me"], [0.4, 89, "ben looking up tombs county"]], "parity": true} |

Profile: phases_ls500_c64.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

