# chatterbox: streaming TTS, tag main

Arm configs: M* PR = python; M* main = python; chatterbox-vllm @8630314 (offline batch) = eager; Chatterbox-TTS-Server @915ae28 (official package, serial) = default

## audio-s/s: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 10.95 [10.86-11.05] n=3 | 14.78 [14.52-14.95] n=3 | 18.71 [17.69-19.45] n=3 | 22.44 [21.74-22.84] n=3 | 27.16 [26.08-28.17] n=3 | 33.94 [33.52-34.18] n=3 | 34.19 [33.49-34.59] n=3 |
| M* main | 9.85 [9.84-9.88] n=3 | 13.47 [13.30-13.68] n=3 | 18.54 [18.19-19.14] n=3 | 21.43 [21.08-21.83] n=3 | 26.44 [26.01-27.15] n=3 | 32.45 [32.25-32.63] n=3 | 32.84 [32.72-32.90] n=3 |
| chatterbox-vllm @8630314 (offline batch) | 3.61 [3.58-3.65] n=3 | - | - | 8.84 [8.78-8.91] n=3 | - | 12.03 [11.91-12.09] n=3 | 13.46 [13.39-13.54] n=3 |
| Chatterbox-TTS-Server @915ae28 (official package, serial) | 2.04 [2.02-2.05] n=3 | 2.08 [2.07-2.09] n=3 | 2.03 [1.95-2.08] n=3 | 2.04 [2.02-2.07] n=3 | - | - | - |

## RTF p50: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.092 [0.090-0.094] n=3 | 0.133 [0.127-0.138] n=3 | 0.207 [0.199-0.216] n=3 | 0.342 [0.336-0.346] n=3 | 0.575 [0.551-0.599] n=3 | 0.913 [0.903-0.929] n=3 | 1.798 [1.776-1.817] n=3 |
| M* main | 0.104 [0.102-0.105] n=3 | 0.144 [0.140-0.150] n=3 | 0.211 [0.206-0.213] n=3 | 0.358 [0.352-0.363] n=3 | 0.586 [0.574-0.604] n=3 | 0.964 [0.956-0.971] n=3 | 1.853 [1.839-1.862] n=3 |
| chatterbox-vllm @8630314 (offline batch) | 0.281 [0.279-0.284] n=3 | - | - | 0.993 [0.985-0.998] n=3 | - | 2.892 [2.876-2.916] n=3 | 5.186 [5.160-5.218] n=3 |
| Chatterbox-TTS-Server @915ae28 (official package, serial) | 0.504 [0.500-0.508] n=3 | 0.918 [0.903-0.942] n=3 | 1.966 [1.880-2.070] n=3 | 3.767 [3.669-3.875] n=3 | - | - | - |

## TTFA p50 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.132 [0.131-0.132] n=3 | 0.191 [0.190-0.192] n=3 | 0.306 [0.300-0.312] n=3 | 0.518 [0.500-0.538] n=3 | 0.834 [0.796-0.886] n=3 | 1.490 [1.489-1.491] n=3 | 5.419 [5.353-5.503] n=3 |
| M* main | 0.146 [0.144-0.147] n=3 | 0.210 [0.208-0.211] n=3 | 0.312 [0.289-0.326] n=3 | 0.539 [0.527-0.560] n=3 | 0.874 [0.842-0.893] n=3 | 1.570 [1.567-1.572] n=3 | 5.546 [5.434-5.606] n=3 |
| chatterbox-vllm @8630314 (offline batch) | 1.299 [1.289-1.310] n=3 | - | - | 4.132 [4.097-4.160] n=3 | - | 12.390 [12.332-12.488] n=3 | 22.416 [22.282-22.540] n=3 |
| Chatterbox-TTS-Server @915ae28 (official package, serial) | 2.082 [2.047-2.123] n=3 | 2.238 [2.123-2.318] n=3 | 2.455 [2.382-2.558] n=3 | 2.288 [2.182-2.350] n=3 | - | - | - |

## TTFA p99 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.134 [0.133-0.135] n=3 | 0.269 [0.249-0.285] n=3 | 0.559 [0.437-0.792] n=3 | 0.699 [0.643-0.737] n=3 | 1.155 [1.039-1.223] n=3 | 2.053 [1.943-2.140] n=3 | 6.515 [6.307-6.660] n=3 |
| M* main | 0.148 [0.148-0.149] n=3 | 0.271 [0.266-0.276] n=3 | 0.427 [0.414-0.442] n=3 | 0.712 [0.707-0.721] n=3 | 1.147 [1.125-1.160] n=3 | 2.115 [2.064-2.147] n=3 | 6.761 [6.699-6.835] n=3 |
| chatterbox-vllm @8630314 (offline batch) | 1.988 [1.964-2.003] n=3 | - | - | 4.465 [4.433-4.491] n=3 | - | 12.869 [12.733-12.947] n=3 | 22.744 [22.688-22.797] n=3 |
| Chatterbox-TTS-Server @915ae28 (official package, serial) | 3.344 [3.228-3.425] n=3 | 4.588 [4.497-4.690] n=3 | 7.890 [7.655-8.255] n=3 | 14.630 [14.405-14.951] n=3 | - | - | - |

## Gate (WER, Whisper large-v3)

Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.14%, same-code spread (two M* PR processes) 0.32 pt, tolerance 1.00 pt; median s/word over arms 0.255

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| cbserver__default | 1.14 | 128 | 0 | 0.261 | PASS | yes |
| cbvllm__eager | 1.55 | 128 | 0 | 0.273 | PASS | yes |
| mstar_main | 1.41 | 128 | 0 | 0.253 | PASS | yes |
| mstar_pr | 1.59 | 128 | 0 | 0.255 | PASS | yes |
| mstar_pr_b | 1.91 | 128 | 0 | 0.255 | PASS | yes |

