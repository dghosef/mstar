# omniasr_7b: Omnilingual ASR CTC 7b (LibriSpeech test-clean, RTFx)



Arms: M* PR = config; M* main = config; omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) = bf16_sorted
Competitor choice: no engine serves the CTC models (vLLM PR #36864 is open and LLM-variant only; nothing in SGLang, SGLang-Omni, vLLM-Omni or transformers), so the official package's own batched pipeline runs as an offline-batch arm on the same v2 weights; batch b in the c column, length-sorted (its strongest offline setting) plus as-shipped input order as a tune knob

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## ls500: LibriSpeech test-clean (500 utterances, fixed stride), RTFx = audio s / wall s

### audio-s/s

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 189.837 [189.678-189.996] n=2 | 515.925 [494.762-537.088] n=2 | 534.955 [533.202-536.709] n=2 |
| M* main | 100.090 [99.907-100.273] n=2 | 264.331 [259.178-269.484] n=2 | 300.968 [300.028-301.908] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 99.310 [99.223-99.396] n=2 | 433.464 [432.803-434.126] n=2 | 490.447 [489.120-491.774] n=2 |

### req/s

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 25.229 [25.208-25.250] n=2 | 69.383 [66.537-72.229] n=2 | 71.942 [71.706-72.178] n=2 |
| M* main | 13.302 [13.277-13.326] n=2 | 35.548 [34.855-36.241] n=2 | 40.475 [40.349-40.601] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 13.198 [13.187-13.210] n=2 | 58.293 [58.204-58.382] n=2 | 65.957 [65.778-66.135] n=2 |

### latency p50 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.037 [0.037-0.038] n=2 | 0.219 [0.210-0.228] n=2 | 0.858 [0.856-0.860] n=2 |
| M* main | 0.073 [0.073-0.073] n=2 | 0.430 [0.426-0.434] n=2 | 1.574 [1.566-1.581] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 0.075 [0.075-0.076] n=2 | 0.220 [0.218-0.222] n=2 | 0.694 [0.694-0.694] n=2 |

### latency p99 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.079 [0.079-0.079] n=2 | 0.410 [0.408-0.412] n=2 | 1.267 [1.256-1.278] n=2 |
| M* main | 0.094 [0.094-0.094] n=2 | 0.765 [0.756-0.774] n=2 | 2.034 [1.998-2.070] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 0.106 [0.098-0.114] n=2 | 1.171 [1.170-1.173] n=2 | 4.595 [4.594-4.595] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.037 [0.037-0.037] n=2 | 0.219 [0.210-0.228] n=2 | 0.858 [0.856-0.860] n=2 |
| M* main | 0.073 [0.073-0.073] n=2 | 0.430 [0.426-0.434] n=2 | 1.573 [1.566-1.581] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | - | - | - |

M* PR audio-s/s relative to each arm:

- M* PR vs M* main: c=1 +89.7%, c=16 +95.2%, c=64 +77.7%
- M* PR vs omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch): c=1 +91.2%, c=16 +19.0%, c=64 +9.1%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_main | PASS | ls500: {"wer": 2.7122226135963365, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "ai mi"], [0.4, 89, "then looking up tombs county"], [0.4, 79, "tomorrow is the examination"]], "parity": true} |
| mstar_pr | PASS | ls500: {"wer": 2.7122226135963365, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "ai mi"], [0.4, 89, "then looking up tombs county"], [0.4, 79, "tomorrow is the examination"]], "parity": true} |
| mstar_pr_b | PASS | ls500: {"wer": 2.7122226135963365, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "ai mi"], [0.4, 89, "then looking up tombs county"], [0.4, 79, "tomorrow is the examination"]], "parity": true} |
| omniasr_official__bf16_sorted | PASS | ls500: {"wer": 2.571327932370553, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "ai mi"], [0.4, 89, "then looking up tombs county"], [0.4, 79, "tomorrow is the examination"]], "parity": true} |
| omniasr_official__bf16_unsorted | PASS | ls500: {"wer": 2.571327932370553, "n": 128, "missing": 0, "failed": 0, "worst": [[1.0, 25, "ai mi"], [0.4, 89, "then looking up tombs county"], [0.4, 79, "tomorrow is the examination"]], "parity": true} |

Profile: phases_ls500_c64.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

