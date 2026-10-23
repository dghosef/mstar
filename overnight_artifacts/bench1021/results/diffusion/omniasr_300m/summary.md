# omniasr_300m: Omnilingual ASR CTC 300m (LibriSpeech test-clean, RTFx)



Arms: M* PR = config; M* main = config; omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) = bf16_sorted
Competitor choice: no engine serves the CTC models (vLLM PR #36864 is open and LLM-variant only; nothing in SGLang, SGLang-Omni, vLLM-Omni or transformers), so the official package's own batched pipeline runs as an offline-batch arm on the same v2 weights; batch b in the c column, length-sorted (its strongest offline setting) plus as-shipped input order as a tune knob

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## ls500: LibriSpeech test-clean (500 utterances, fixed stride), RTFx = audio s / wall s

### audio-s/s

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 297.855 [297.705-298.005] n=2 | 1924.098 [1896.786-1951.410] n=2 | 1753.793 [1735.423-1772.163] n=2 |
| M* main | 255.410 [254.442-256.379] n=2 | 1482.753 [1469.516-1495.990] n=2 | 1799.769 [1782.072-1817.465] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 237.154 [235.286-239.022] n=2 | 1341.416 [1318.058-1364.774] n=2 | 1620.942 [1614.708-1627.176] n=2 |

### req/s

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 39.584 [39.564-39.604] n=2 | 258.758 [255.085-262.431] n=2 | 235.855 [233.384-238.325] n=2 |
| M* main | 33.944 [33.815-34.072] n=2 | 199.404 [197.624-201.185] n=2 | 242.037 [239.658-244.417] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 31.517 [31.269-31.766] n=2 | 180.397 [177.256-183.538] n=2 | 217.988 [217.150-218.827] n=2 |

### latency p50 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.027 [0.027-0.027] n=2 | 0.060 [0.060-0.060] n=2 | 0.270 [0.267-0.273] n=2 |
| M* main | 0.029 [0.027-0.030] n=2 | 0.076 [0.076-0.076] n=2 | 0.261 [0.256-0.265] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 0.033 [0.032-0.033] n=2 | 0.075 [0.074-0.075] n=2 | 0.227 [0.226-0.228] n=2 |

### latency p99 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.036 [0.034-0.038] n=2 | 0.094 [0.093-0.095] n=2 | 0.318 [0.314-0.322] n=2 |
| M* main | 0.041 [0.039-0.043] n=2 | 0.138 [0.130-0.145] n=2 | 0.378 [0.373-0.383] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | 0.045 [0.045-0.046] n=2 | 0.283 [0.283-0.283] n=2 | 1.047 [1.047-1.048] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=16 | c=64 |
|---|---:|---:|---:|
| M* PR | 0.027 [0.027-0.027] n=2 | 0.060 [0.060-0.060] n=2 | 0.270 [0.267-0.273] n=2 |
| M* main | 0.029 [0.027-0.030] n=2 | 0.076 [0.076-0.076] n=2 | 0.261 [0.256-0.265] n=2 |
| omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch) | - | - | - |

M* PR audio-s/s relative to each arm:

- M* PR vs M* main: c=1 +16.6%, c=16 +29.8%, c=64 -2.6%
- M* PR vs omnilingual-asr 0.2.0 ASRInferencePipeline (official) (offline batch): c=1 +25.6%, c=16 +43.4%, c=64 +8.2%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| mstar_main | PASS | ls500: {"wer": 5.001761183515322, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "i me"], [0.3333333333333333, 107, "money miron's twite told peas put it down to warm"]], "parity": true} |
| mstar_pr | PASS | ls500: {"wer": 4.966537513208876, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "i me"], [0.3333333333333333, 107, "money miron's twite told peas put it down to warm"]], "parity": true} |
| mstar_pr_b | PASS | ls500: {"wer": 4.931313842902431, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "i me"], [0.3333333333333333, 107, "money miron's twite told peas put it down to warm"]], "parity": true} |
| omniasr_official__bf16_sorted | PASS | ls500: {"wer": 4.966537513208876, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "i me"], [0.3333333333333333, 107, "money miron's twite told peas put it down to warm"]], "parity": true} |
| omniasr_official__bf16_unsorted | PASS | ls500: {"wer": 5.001761183515322, "n": 128, "missing": 0, "failed": 0, "worst": [[0.6, 79, "tomorrow as the examination"], [0.5, 25, "i me"], [0.3333333333333333, 107, "money miron's twite told peas put it down to warm"]], "parity": true} |

Profile: phases_ls500_c64.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

