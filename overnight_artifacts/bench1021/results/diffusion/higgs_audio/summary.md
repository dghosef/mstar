# higgs_audio: Higgs Audio v3 STT (LibriSpeech test-clean)



Arms: M* PR = config; M* main = config; bosonai higgs-audio-v3-stt official remote code (transcribe_batch) (offline batch) = eager
Competitor choice: nothing serves the HiggsAudio3Model STT architecture (vLLM: no Higgs class; vLLM-Omni and SGLang-Omni serve only the Higgs TTS models; Boson's vLLM container is closed), so the model card's official remote code (transformers 4.51.0, bf16, the repo's batched transcribe_batch) runs as an offline-batch arm: default prompt, enable_thinking=False (M*'s default), greedy, 1024 max new tokens, 4 s chunks; attention eager (the card's) vs sdpa tuned

Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no server, queueing or HTTP cost).

## ls500: LibriSpeech test-clean (500 utterances, fixed stride), RTFx = audio s / wall s

### audio-s/s

| arm | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| M* PR | 111.126 [110.646-111.607] n=2 | 433.630 [430.641-436.619] n=2 | 795.268 [786.793-803.742] n=2 |
| M* main | 75.616 [75.362-75.869] n=2 | 194.251 [174.249-214.253] n=2 | 402.116 [396.485-407.747] n=2 |
| bosonai higgs-audio-v3-stt official remote code (transcribe_batch) (offline batch) | 4.937 [4.933-4.940] n=2 | 34.050 [34.030-34.071] n=2 | 123.326 [123.055-123.596] n=2 |

### req/s

| arm | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| M* PR | 14.154 [14.092-14.215] n=2 | 60.130 [59.716-60.545] n=2 | 106.474 [105.339-107.608] n=2 |
| M* main | 9.631 [9.599-9.663] n=2 | 26.936 [24.163-29.710] n=2 | 53.837 [53.083-54.591] n=2 |
| bosonai higgs-audio-v3-stt official remote code (transcribe_batch) (offline batch) | 0.629 [0.628-0.629] n=2 | 4.722 [4.719-4.725] n=2 | 16.511 [16.475-16.548] n=2 |

### latency p50 (s)

| arm | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| M* PR | 0.058 [0.058-0.059] n=2 | 0.115 [0.114-0.116] n=2 | 0.263 [0.262-0.264] n=2 |
| M* main | 0.089 [0.089-0.089] n=2 | 0.225 [0.221-0.228] n=2 | 0.502 [0.501-0.502] n=2 |
| bosonai higgs-audio-v3-stt official remote code (transcribe_batch) (offline batch) | 1.478 [1.478-1.478] n=2 | 1.568 [1.567-1.570] n=2 | 1.760 [1.740-1.781] n=2 |

### latency p99 (s)

| arm | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| M* PR | 0.165 [0.162-0.167] n=2 | 0.342 [0.340-0.345] n=2 | 0.773 [0.764-0.782] n=2 |
| M* main | 0.399 [0.399-0.399] n=2 | 1.588 [0.949-2.227] n=2 | 1.571 [1.529-1.612] n=2 |
| bosonai higgs-audio-v3-stt official remote code (transcribe_batch) (offline batch) | 2.564 [2.546-2.583] n=2 | 3.666 [3.666-3.666] n=2 | 4.234 [4.220-4.248] n=2 |

### TTFB p50 (s)

| arm | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| M* PR | 0.058 [0.058-0.059] n=2 | 0.115 [0.114-0.116] n=2 | 0.263 [0.262-0.264] n=2 |
| M* main | 0.089 [0.089-0.089] n=2 | 0.225 [0.221-0.228] n=2 | 0.502 [0.501-0.502] n=2 |
| bosonai higgs-audio-v3-stt official remote code (transcribe_batch) (offline batch) | - | - | - |

M* PR audio-s/s relative to each arm:

- M* PR vs M* main: c=1 +47.0%, c=8 +123.2%, c=32 +97.8%
- M* PR vs bosonai higgs-audio-v3-stt official remote code (transcribe_batch) (offline batch): c=1 +2151.0%, c=8 +1173.5%, c=32 +544.9%

## Sanity gate (dgen_gate.py; parity annotated, not gating)

| tag | verdict | detail |
|---|---|---|
| higgs_official__eager | PASS | ls500: {"wer": 1.6555125044029588, "n": 128, "missing": 0, "failed": 0, "worst": [[0.5, 25, "i me"], [0.2, 89, "been looking up toombs county"], [0.16666666666666666, 107, "money my iron's twite told please put it down to warm"]], "parity": true} |
| mstar_main | PASS | ls500: {"wer": 1.585065163790067, "n": 128, "missing": 0, "failed": 0, "worst": [[0.2, 89, "been looking up toombs county"], [0.16666666666666666, 107, "money my iron's twite told please put it down to warm"], [0.16666666666666666, 66, "will the hirans hear his groans"]], "parity": true} |
| mstar_pr | PASS | ls500: {"wer": 1.6555125044029588, "n": 128, "missing": 0, "failed": 0, "worst": [[0.5, 25, "i me"], [0.2, 89, "been looking up toombs county"], [0.16666666666666666, 107, "money my iron's twite told please put it down to warm"]], "parity": true} |
| mstar_pr_b | PASS | ls500: {"wer": 1.6555125044029588, "n": 128, "missing": 0, "failed": 0, "worst": [[0.5, 25, "i me"], [0.2, 89, "been looking up toombs county"], [0.16666666666666666, 107, "money my iron's twite told please put it down to warm"]], "parity": true} |

Profile: phases_ls500_c32.txt (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)

