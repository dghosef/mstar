# TTS lane on node2: M* vs vllm-omni, SGLang-Omni, vox-serve, fish-speech

_Generated 2026-10-23 03:59 UTC by report_tts.py._

Hardware: node2, 8x H200 (see REPORT_DIFFUSION.md); every timed server on GPU 0 alone. Arms: M* PR d248477ec and M* main abdfd5802 (own configs, Rust graph runtime, MSTAR_SHM_ARENA=1), each competitor on its best tuned knob set (tune table below), pinned versions in each model's versions/. Text: 256 LibriSpeech test-clean sentences (8-30 words, every ~7th utterance in id order; mean 17 words), one streaming client for every engine (dtts_client.py): TTFA = first audio bytes, RTF = e2e / audio seconds, audio-s/s = total audio / wall. Closed loop at c = 1..64, 1 warmup wave per point after an untimed 64-sentence pass, 3 rounds with fresh servers and rotated order; a point counts only with failed == 0. Sampling: each model's default as M* serves it, matched where the engine exposes it (Fish: temp 0.8, top_p 0.8, top_k 30, rep 1.0 sent to SGLang-Omni and fish-speech; vllm-omni's Fish deploy YAML copied with top_p 0.8). Gate: Whisper large-v3 WER over 128 sentences per arm (criteria in dtts.py), seconds-per-word sanity, no silent/runaway output.

## Qwen3-TTS 1.7B CustomVoice (voice vivian, English, 24 kHz)

Phases completed: collect, gate, profile, rounds, select, setup, summarize, tune.

Tuning (score = geometric-mean audio-s/s over the tune points):

| engine | knob set | per-point audio-s/s | TTFA p50 c=1 | score | chosen |
|---|---|---|---:|---:|---|
| vllmomni | default | c=1: 12.6, c=32: 118.4 | 0.029 | 38.68 |  |
| vllmomni | high_conc | c=1: 12.7, c=32: 131.8 | 0.029 | 40.88 | yes |
| sgomni | base | c=1: 17.6, c=32: 245.1 | 0.013 | 65.69 |  |
| sgomni | mrr128 | c=1: 17.6, c=32: 246.5 | 0.013 | 65.96 | yes |

Arm configs: M* PR = python; M* main = python; vllm-omni 0.30.0 = high_conc; SGLang-Omni 0.1.9 = mrr128

## audio-s/s: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 15.61 [15.58-15.64] n=3 | 22.20 [22.12-22.27] n=3 | 39.02 [38.36-39.58] n=3 | 64.11 [63.60-65.02] n=3 | 97.13 [94.94-99.54] n=3 | 138.54 [135.42-140.13] n=3 | 190.88 [187.76-193.40] n=3 |
| M* main | 14.12 [14.08-14.16] n=3 | 20.83 [20.77-20.96] n=3 | 37.42 [37.12-37.96] n=3 | 62.23 [61.15-63.88] n=3 | 93.56 [92.70-95.01] n=3 | 123.01 [110.47-129.47] n=3 | 145.82 [140.94-148.69] n=3 |
| vllm-omni 0.30.0 | 12.70 [12.68-12.72] n=3 | 22.19 [21.95-22.38] n=3 | 40.41 [39.65-40.82] n=3 | 66.32 [65.29-67.83] n=3 | 97.59 [96.15-98.39] n=3 | 133.24 [131.59-135.26] n=3 | 170.80 [170.48-171.11] n=3 |
| SGLang-Omni 0.1.9 | 17.66 [17.60-17.71] n=3 | 31.58 [31.30-31.85] n=3 | 59.23 [58.29-60.42] n=3 | 103.96 [102.35-106.92] n=3 | 165.75 [164.23-168.16] n=3 | 243.40 [238.59-246.07] n=3 | 346.23 [343.78-347.85] n=3 |

## RTF p50: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.064 [0.064-0.064] n=3 | 0.089 [0.088-0.089] n=3 | 0.099 [0.099-0.099] n=3 | 0.117 [0.116-0.117] n=3 | 0.152 [0.151-0.152] n=3 | 0.216 [0.215-0.216] n=3 | 0.314 [0.313-0.316] n=3 |
| M* main | 0.071 [0.070-0.071] n=3 | 0.093 [0.092-0.094] n=3 | 0.103 [0.102-0.104] n=3 | 0.121 [0.120-0.122] n=3 | 0.158 [0.157-0.159] n=3 | 0.224 [0.221-0.225] n=3 | 0.433 [0.427-0.441] n=3 |
| vllm-omni 0.30.0 | 0.079 [0.078-0.079] n=3 | 0.086 [0.086-0.086] n=3 | 0.096 [0.095-0.097] n=3 | 0.113 [0.113-0.114] n=3 | 0.150 [0.149-0.151] n=3 | 0.222 [0.220-0.227] n=3 | 0.345 [0.343-0.349] n=3 |
| SGLang-Omni 0.1.9 | 0.056 [0.056-0.057] n=3 | 0.060 [0.060-0.061] n=3 | 0.065 [0.065-0.065] n=3 | 0.072 [0.072-0.072] n=3 | 0.086 [0.085-0.087] n=3 | 0.115 [0.114-0.116] n=3 | 0.171 [0.169-0.173] n=3 |

## TTFA p50 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.018 [0.018-0.019] n=3 | 0.030 [0.030-0.031] n=3 | 0.038 [0.037-0.038] n=3 | 0.041 [0.041-0.041] n=3 | 0.046 [0.044-0.047] n=3 | 0.052 [0.048-0.054] n=3 | 0.070 [0.070-0.071] n=3 |
| M* main | 0.019 [0.018-0.019] n=3 | 0.033 [0.032-0.034] n=3 | 0.039 [0.038-0.039] n=3 | 0.044 [0.042-0.045] n=3 | 0.055 [0.054-0.056] n=3 | 0.075 [0.073-0.078] n=3 | 0.085 [0.083-0.086] n=3 |
| vllm-omni 0.30.0 | 0.029 [0.028-0.029] n=3 | 0.034 [0.033-0.035] n=3 | 0.039 [0.038-0.042] n=3 | 0.050 [0.049-0.051] n=3 | 0.066 [0.063-0.068] n=3 | 0.102 [0.100-0.105] n=3 | 0.151 [0.147-0.157] n=3 |
| SGLang-Omni 0.1.9 | 0.013 [0.013-0.013] n=3 | 0.015 [0.014-0.016] n=3 | 0.016 [0.016-0.016] n=3 | 0.018 [0.018-0.018] n=3 | 0.020 [0.020-0.020] n=3 | 0.023 [0.022-0.023] n=3 | 0.027 [0.027-0.027] n=3 |

## TTFA p99 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.020 [0.020-0.020] n=3 | 0.040 [0.039-0.041] n=3 | 0.050 [0.047-0.052] n=3 | 0.063 [0.055-0.069] n=3 | 0.083 [0.082-0.084] n=3 | 0.110 [0.108-0.113] n=3 | 0.170 [0.163-0.176] n=3 |
| M* main | 0.020 [0.020-0.021] n=3 | 0.041 [0.040-0.042] n=3 | 0.047 [0.046-0.050] n=3 | 0.065 [0.059-0.070] n=3 | 0.090 [0.084-0.097] n=3 | 0.123 [0.111-0.129] n=3 | 0.261 [0.181-0.318] n=3 |
| vllm-omni 0.30.0 | 0.030 [0.030-0.030] n=3 | 0.046 [0.045-0.047] n=3 | 0.056 [0.050-0.060] n=3 | 0.071 [0.069-0.075] n=3 | 0.111 [0.098-0.131] n=3 | 0.168 [0.156-0.180] n=3 | 0.254 [0.237-0.280] n=3 |
| SGLang-Omni 0.1.9 | 0.016 [0.015-0.016] n=3 | 0.020 [0.017-0.024] n=3 | 0.025 [0.024-0.026] n=3 | 0.032 [0.032-0.033] n=3 | 0.045 [0.044-0.046] n=3 | 0.075 [0.074-0.076] n=3 | 0.127 [0.123-0.130] n=3 |

## Gate (WER, Whisper large-v3)

Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.37%, same-code spread (two M* PR processes) 0.46 pt, tolerance 1.00 pt; median s/word over arms 0.409

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| mstar_main | 1.82 | 128 | 0 | 0.411 | PASS | yes |
| mstar_pr | 1.37 | 128 | 0 | 0.409 | PASS | yes |
| mstar_pr_b | 1.82 | 128 | 0 | 0.414 | PASS | yes |
| sgomni__base | 1.96 | 128 | 0 | 0.400 | PASS | yes |
| sgomni__mrr128 | 1.96 | 128 | 0 | 0.400 | PASS | yes |
| vllmomni__default | 1.73 | 128 | 0 | 0.413 | PASS | yes |
| vllmomni__high_conc | 1.55 | 128 | 0 | 0.409 | PASS | yes |

Server commands (round 1):

```
/root/work/mstar-main/.venv/bin/python -m mstar.cli.main serve qwen3_tts_1p7b --config /root/work/mstar-main/configs/qwen3tts_1p7b.yaml --gpus 0 --port 8400
/root/work/mstar/.venv/bin/python -m mstar.cli.main serve qwen3_tts_1p7b --config /root/work/mstar/configs/qwen3tts_1p7b.yaml --gpus 0 --port 8400
/root/work/engines/sgomni/bin/sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --host 127.0.0.1 --port 8400 --config /root/work/dbench/tts_configs/sgomni_qwen3tts_1p7b.yaml --tts_engine.engine.max_running_requests 128 --tts_engine.engine.max_queued_requests 512
/root/work/engines/vllmomni/bin/vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --host 127.0.0.1 --port 8400 --deploy-config /root/work/engines/vllmomni/lib/python3.12/site-packages/vllm_omni/deploy/qwen3_tts_high_concurrency_mrv2_single_gpu.yaml
```

<details><summary>notes</summary>

```
notes for qwen3tts_1p7b
dropped engines: {}
startup s: vllmomni: median 84 n=6; sgomni: median 96 n=6; mstar-pr: median 140 n=8; mstar-main: median 176 n=4
main_mstar-main_python_r1: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-main_python_r2: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-main_python_r3: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r1: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r2: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r3: failed captures 0, graph runtime rust True, rust frontend False
```

</details>

<details><summary>profile phases_c1</summary>

```

=== worker_0  iters 100-1700  records=17  batch=1.00 (range 1.0-1.0) ===
    work: 1,700 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    0.029    0.000    0.510     1699   28764.7 
engine.admit                                 0.011    0.010    0.020     1579   10364.7 
engine.collect_outputs                       0.099    0.090    0.480     1699   99352.9 
engine.commit                                0.089    0.080    0.290     1578   82852.9 
engine.declare_step                          0.019    0.020    0.030     1699   18811.8 
engine.forward                               0.127    0.100    1.710     1699  126994.1 
engine.plan.fresh                            2.385    1.430   25.150       23   32264.7 
engine.plan.promoted                         0.161    0.150    0.230     1556  146994.1 
engine.postprocess                           0.011    0.010    0.080     1699   11170.6 
engine.pre_admit                             0.021    0.020    0.030     1556   18817.6 
engine.pre_declare                           0.020    0.020    0.020     1556   18305.9 
engine.preprocess                            0.109    0.100    0.150     1699  109352.9 
gpu_exec                                     4.584    4.570    5.140     1698 4578941.2 
iter_total                                   4.455    4.570    6.500     1700 4454705.9 
res.admit.code_predictor                     0.000    0.000    0.010      359       5.9 
res.admit.talker_attn                        0.000    0.000    0.000      287       0.0 
res.admit.talker_kv                          0.011    0.030    0.030      287    1835.3 
res.admit.talker_pos                         0.000    0.000    0.000      287       0.0 
res.admit.talker_sampler                     0.000    0.000    0.000      287       0.0 
res.commit.code_predictor                    0.030    0.020    0.100     1579   28288.2 
res.commit.sampler.offset                    0.021    0.020    0.030     3158   38129.4*
res.commit.talker_attn                       0.000    0.000    0.000     1578       0.0 
res.commit.talker_kv                         0.010    0.010    0.010     1579    9288.2 
res.commit.talker_pos                        0.010    0.010    0.010     1578    9282.4 
res.commit.talker_sampler                    0.041    0.040    0.080     1578   38211.8 
res.plan.code_predictor                      0.079    0.060    0.290     1915   88864.7*
res.plan.talker_attn                         0.668    0.000    3.530     1843  723852.9*
res.plan.talker_kv                           0.082    0.010    0.610     1843   88705.9*
res.plan.talker_pos                          0.012    0.000    0.130     1843   13058.8*
res.plan.talker_sampler                      0.086    0.070    0.230     1843   93747.1*
res.pre_plan.code_predictor                  0.011    0.010    0.010     1556   10188.2 
res.pre_plan.talker_attn                     0.146    0.130    0.230     1556  134023.5 
res.pre_plan.talker_kv                       0.212    0.200    0.380     1556  194417.6 
res.pre_plan.talker_pos                      0.038    0.030    0.050     1556   34994.1 
res.pre_plan.talker_sampler                  0.008    0.000    0.010     1556    7552.9 
rust.send_outputs.call                       0.010    0.010    0.050     1682    9894.1 
rust.send_outputs.marshal                    0.008    0.000    0.090     1682    8129.4 
speculate                                    0.069    0.070    0.110     1476   60223.5 
submit_spec                                  1.090    0.840    4.430     1642 1052352.9 
worker.await_admit_settled                   1.766    0.580   28.340       60   62317.6 
worker.gpu_thread.await_plan                 0.276    0.260    0.450     1642  266547.1 
worker.gpu_thread.exec                       0.730    0.660    2.510     1699  729594.1 
worker.gpu_thread.prepare_inputs             0.256    0.030    4.120     1699  256323.5 
worker.plan_thread.await_commit              0.005    0.000    0.010     1556    4235.3 
worker.plan_thread.pre_plan                  0.486    0.450    0.800     1556  444594.1 
worker.plan_thread.reserve_slot              0.022    0.020    0.040     1556   20417.6 
worker.postprocess.check_stop                0.037    0.030    0.140     1682   36635.3 
worker.postprocess.check_stop_tail           0.035    0.030    0.140     1682   34335.3 
worker.postprocess.cleanup_inputs            0.010    0.010    0.020     1699    9994.1 
worker.postprocess.completion_event_sync     2.894    3.170    3.990     1682 2863794.1 
worker.postprocess.event_sync                2.900    3.160    4.080     1699 2898394.1 
worker.postprocess.pending_loop_stops        0.010    0.010    0.010     1682    9894.1 
worker.postprocess.prematerialize            0.012    0.000    0.140     1682   12200.0 
worker.postprocess.register_outputs          0.012    0.000    0.180     1682   12211.8 
worker.postprocess.route                     0.025    0.020    0.040     1682   25011.8 
worker.postprocess.route_outputs             0.085    0.080    0.150     1682   83794.1 
worker.postprocess.send                      0.022    0.010    0.140     1682   21517.6 
worker.postprocess.send_outputs              0.077    0.060    0.840     1682   76164.7 
worker.postprocess.send_prep                 0.010    0.010    0.010     1682    9894.1 
worker.postprocess.stop_loops                0.000    0.000    0.000     1682       0.0 
worker.postprocess.store_tensors             0.050    0.050    0.090     1682   49470.6 
worker.postprocess.update_lru                0.000    0.000    0.000     1682       0.0 
worker.postprocess_batch                     3.151    3.370    4.550     1699 3148776.5 
worker.
```

</details>

<details><summary>profile phases_c64</summary>

```

=== worker_0  iters 100-100  records=1  batch=37.19 (range 37.2-37.2) ===
    work: 3,719 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    0.290    0.000    1.660       99    7719.8 
engine.admit                                 0.030    0.020    0.070       69     556.6 
engine.collect_outputs                       0.500    0.320    1.710       99   13310.0 
engine.commit                                0.370    0.280    1.020       69    6864.7 
engine.declare_step                          0.030    0.040    0.050      100     806.7 
engine.forward                               0.860    0.190    2.920      100   23124.5 
engine.plan.fresh                            3.290    1.700   24.950       16   14154.3 
engine.plan.promoted                         0.320    0.310    0.370       53    4560.4 
engine.postprocess                           0.090    0.080    0.210       99    2395.8 
engine.pre_admit                             0.110    0.100    0.130       53    1567.6 
engine.pre_declare                           0.040    0.040    0.050       53     570.0 
engine.preprocess                            0.240    0.200    0.660      100    6453.3 
gpu_exec                                     8.790    7.210   24.710       98  231626.8 
iter_total                                  10.520    7.440   34.220      100  282871.7 
res.admit.code_predictor                     0.000    0.000    0.000      352       0.0*
res.admit.talker_attn                        0.000    0.000    0.000      280       0.0*
res.admit.talker_kv                          0.010    0.010    0.040      280     752.9*
res.admit.talker_pos                         0.000    0.000    0.000      280       0.0*
res.admit.talker_sampler                     0.000    0.000    0.000      280       0.0*
res.commit.code_predictor                    0.110    0.030    0.590       69    2040.9 
res.commit.sampler.offset                    0.070    0.020    0.370      138    2597.5*
res.commit.talker_attn                       0.000    0.000    0.000       69       0.0 
res.commit.talker_kv                         0.050    0.060    0.060       69     927.7 
res.commit.talker_pos                        0.030    0.030    0.040       69     556.6 
res.commit.talker_sampler                    0.180    0.150    0.400       69    3339.6 
res.plan.code_predictor                      0.130    0.080    0.310      405   14157.0*
res.plan.talker_attn                        43.770    0.290    3.510      333 3919174.5*
res.plan.talker_kv                           0.430    0.520    0.620      333   38502.3*
res.plan.talker_pos                          0.060    0.060    0.130      333    5372.4*
res.plan.talker_sampler                      0.190    0.050    0.270      333   17012.6*
res.pre_plan.code_predictor                  0.230    0.200    0.950       53    3277.8 
res.pre_plan.talker_attn                     0.220    0.180    0.330       53    3135.3 
res.pre_plan.talker_kv                       0.300    0.270    0.530       53    4275.3 
res.pre_plan.talker_pos                      0.080    0.070    0.090       53    1140.1 
res.pre_plan.talker_sampler                  0.090    0.130    0.310       53    1282.6 
rust.send_outputs.call                       0.080    0.050    0.250       99    2129.6 
rust.send_outputs.marshal                    0.090    0.050    0.410       99    2395.8 
speculate                                    0.300    0.280    0.450       35    2823.3 
submit_spec                                  2.920    2.340    5.110       99   77730.6 
worker.await_admit_settled                   3.170    0.940   28.100       14   11933.3 
worker.gpu_thread.await_plan                 0.250    0.010    0.830       99    6655.0 
worker.gpu_thread.exec                       2.900    1.720    7.890       99   77198.2 
worker.gpu_thread.prepare_inputs             2.940    0.600    8.300      100   79053.5 
worker.plan_thread.await_commit              0.120    0.000    0.870       68    2194.1 
worker.plan_thread.pre_plan                  1.110    0.960    2.070       53   15818.8 
worker.plan_thread.reserve_slot              0.020    0.020    0.030       68     365.7 
worker.postprocess.check_stop                0.300    0.130    1.650       99    7986.0 
worker.postprocess.check_stop_tail           0.100    0.140    0.180       99    2662.0 
worker.postprocess.cleanup_inputs            0.050    0.070    0.080       99    1331.0 
worker.postprocess.completion_event_sync     2.780    1.220   11.760       99   74003.8 
worker.postprocess.event_sync                2.770    1.210   11.750       99   73737.6 
worker.postprocess.pending_loop_stops        0.010    0.010    0.020       99     266.2 
worker.postprocess.prematerialize            0.220    0.000    1.640       99    5856.4 
worker.postprocess.register_outputs          0.170    0.000    1.160       99    4525.4 
worker.postprocess.route                     0.100    0.130    0.170       99    2662.0 
worker.postprocess.route_outputs             0.340    0.470    0.620       99    9050.8 
worker.postprocess.send                      0.180    0.090    0.680       99    4791.6 
worker.postprocess.send_outputs              0.930    0.760    3.120       99   24756.7 
worker.postprocess.send_prep                 0.050    0.030    0.180       99    1331.0 
worker.postprocess.stop_loops                0.010    0.000    0.060       99     266.2 
worker.postprocess.store_tensors             0.220    0.300    0.380       99    5856.4 
worker.postprocess.update_lru                0.010    0.010    0.010       99     266.2 
worker.postprocess_batch                     4.640    3.550   12.740       99  123517.1 
worker
```

</details>

## Fish S2-Pro (no reference voice, 44.1 kHz)

Phases completed: collect, gate, profile, rounds, select, setup, summarize, tune.

Engines dropped at setup: {"fishspeech": "smoke failed"}

Tuning (score = geometric-mean audio-s/s over the tune points):

| engine | knob set | per-point audio-s/s | TTFA p50 c=1 | score | chosen |
|---|---|---|---:|---:|---|
| sgomni | base | c=1: 5.1, c=32: 40.5 | 0.481 | 14.44 |  |
| sgomni | mrr128 | c=1: 5.4, c=32: 42.2 | 0.459 | 15.09 | yes |
| vllmomni | high_conc | c=1: 3.8, c=32: 6.3 | 0.077 | 4.89 | yes |
| vllmomni | default | c=1: 3.7, c=32: 5.3 | 0.083 | 4.43 |  |

Arm configs: M* PR = python; M* main = python; SGLang-Omni 0.1.9 = mrr128; vllm-omni 0.30.0 = high_conc

## audio-s/s: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 6.51 [6.50-6.53] n=3 | 10.18 [10.09-10.30] n=3 | 17.15 [16.80-17.33] n=3 | 26.41 [25.18-27.26] n=3 | 41.47 [41.16-41.63] n=3 | 57.70 [48.37-62.60] n=3 | 88.59 [86.06-90.30] n=3 |
| M* main | 4.45 [4.44-4.47] n=3 | 7.35 [7.28-7.41] n=3 | 11.97 [11.72-12.38] n=3 | 17.66 [17.45-17.86] n=3 | 22.11 [19.48-24.34] n=3 | 31.01 [28.07-32.76] n=3 | 35.05 [32.67-36.28] n=3 |
| SGLang-Omni 0.1.9 | 5.21 [5.09-5.37] n=3 | 8.00 [7.85-8.13] n=3 | 13.65 [13.53-13.77] n=3 | 22.06 [21.66-22.57] n=3 | FAILED (n=3) | 41.09 [40.63-41.55] n=2 | 50.01 [49.98-50.03] n=2 |
| vllm-omni 0.30.0 | 3.77 [3.69-3.81] n=3 | 4.61 [4.58-4.62] n=3 | 5.39 [5.38-5.40] n=3 | 5.86 [5.85-5.87] n=3 | 6.15 [6.14-6.15] n=3 | 6.34 [6.34-6.36] n=3 | 6.40 [6.29-6.59] n=3 |

## RTF p50: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.154 [0.153-0.156] n=3 | 0.192 [0.191-0.193] n=3 | 0.224 [0.223-0.225] n=3 | 0.278 [0.277-0.278] n=3 | 0.359 [0.358-0.360] n=3 | 0.468 [0.467-0.469] n=3 | 0.662 [0.656-0.668] n=3 |
| M* main | 0.226 [0.224-0.227] n=3 | 0.266 [0.262-0.270] n=3 | 0.324 [0.319-0.329] n=3 | 0.431 [0.427-0.434] n=3 | 0.611 [0.603-0.622] n=3 | 0.949 [0.936-0.966] n=3 | 1.701 [1.657-1.725] n=3 |
| SGLang-Omni 0.1.9 | 0.192 [0.186-0.196] n=3 | 0.248 [0.241-0.254] n=3 | 0.280 [0.274-0.284] n=3 | 0.341 [0.337-0.345] n=3 | FAILED (n=3) | 0.742 [0.734-0.749] n=2 | 1.246 [1.237-1.256] n=2 |
| vllm-omni 0.30.0 | 0.264 [0.263-0.267] n=3 | 0.425 [0.422-0.426] n=3 | 0.735 [0.735-0.736] n=3 | 1.349 [1.347-1.352] n=3 | 2.569 [2.565-2.573] n=3 | 4.999 [4.989-5.007] n=3 | 9.682 [9.625-9.718] n=3 |

## TTFA p50 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.058 [0.058-0.059] n=3 | 0.081 [0.078-0.084] n=3 | 0.096 [0.092-0.102] n=3 | 0.113 [0.110-0.115] n=3 | 0.136 [0.133-0.140] n=3 | 0.173 [0.172-0.175] n=3 | 0.237 [0.234-0.240] n=3 |
| M* main | 0.112 [0.111-0.114] n=3 | 0.142 [0.141-0.145] n=3 | 0.161 [0.155-0.170] n=3 | 0.198 [0.185-0.214] n=3 | 0.249 [0.236-0.259] n=3 | 0.373 [0.366-0.378] n=3 | 0.570 [0.513-0.664] n=3 |
| SGLang-Omni 0.1.9 | 0.472 [0.457-0.484] n=3 | 0.527 [0.510-0.537] n=3 | 0.591 [0.576-0.600] n=3 | 0.710 [0.695-0.727] n=3 | FAILED (n=3) | 1.477 [1.476-1.479] n=2 | 2.490 [2.473-2.506] n=2 |
| vllm-omni 0.30.0 | 0.077 [0.077-0.078] n=3 | 0.124 [0.123-0.125] n=3 | 0.194 [0.193-0.194] n=3 | 0.359 [0.354-0.363] n=3 | 0.705 [0.697-0.712] n=3 | 1.365 [1.361-1.366] n=3 | 2.640 [2.623-2.662] n=3 |

## TTFA p99 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.060 [0.059-0.061] n=3 | 0.097 [0.096-0.098] n=3 | 0.125 [0.121-0.133] n=3 | 0.146 [0.141-0.153] n=3 | 0.177 [0.166-0.190] n=3 | 0.245 [0.223-0.264] n=3 | 0.347 [0.345-0.348] n=3 |
| M* main | 0.117 [0.116-0.119] n=3 | 0.200 [0.196-0.206] n=3 | 0.234 [0.225-0.247] n=3 | 0.543 [0.505-0.562] n=3 | 0.521 [0.401-0.596] n=3 | 0.621 [0.618-0.626] n=3 | 2.725 [2.549-2.945] n=3 |
| SGLang-Omni 0.1.9 | 0.486 [0.476-0.492] n=3 | 0.628 [0.605-0.645] n=3 | 0.760 [0.738-0.781] n=3 | 0.926 [0.888-0.950] n=3 | FAILED (n=3) | 1.722 [1.695-1.750] n=2 | 2.811 [2.759-2.863] n=2 |
| vllm-omni 0.30.0 | 0.083 [0.078-0.090] n=3 | 0.183 [0.129-0.283] n=3 | 0.230 [0.228-0.232] n=3 | 0.517 [0.389-0.624] n=3 | 0.898 [0.748-1.106] n=3 | 1.673 [1.626-1.702] n=3 | 3.054 [3.041-3.062] n=3 |

## Gate (WER, Whisper large-v3)

Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.14%, same-code spread (two M* PR processes) 0.05 pt, tolerance 1.00 pt; median s/word over arms 0.357

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| mstar_pr | 1.64 | 128 | 0 | 0.361 | PASS | yes |
| mstar_pr_b | 1.59 | 128 | 0 | 0.358 | PASS | yes |
| sgomni__base | 1.50 | 128 | 0 | 0.358 | PASS | yes |
| sgomni__mrr128 | 1.64 | 128 | 0 | 0.355 | PASS | yes |
| vllmomni__default | 1.14 | 128 | 0 | 0.339 | PASS | yes |
| vllmomni__high_conc | 1.23 | 128 | 0 | 0.336 | PASS | yes |

Server commands (round 1):

```
/root/work/mstar-main/.venv/bin/python -m mstar.cli.main serve fish_s2_pro --config /root/work/mstar-main/configs/fish_s2_pro.yaml --gpus 0 --port 8400
/root/work/mstar/.venv/bin/python -m mstar.cli.main serve fish_s2_pro --config /root/work/mstar/configs/fish_s2_pro.yaml --gpus 0 --port 8400
/root/work/engines/sgomni/bin/sgl-omni serve --model-path fishaudio/s2-pro --host 127.0.0.1 --port 8400 --config /root/work/dbench/tts_configs/sgomni_fish_s2pro.yaml --tts_engine.engine.max_running_requests 128 --tts_engine.engine.max_queued_requests 512
/root/work/engines/vllmomni/bin/vllm serve fishaudio/s2-pro --omni --host 127.0.0.1 --port 8400 --deploy-config /root/work/dbench/tts_configs/vllmomni_fish_hc_topp08.yaml
```

<details><summary>notes</summary>

```
notes for fish_s2pro
dropped engines: {"fishspeech": "smoke failed"}
startup s: sgomni: median 48 n=8; vllmomni: median 65 n=6; fishspeech: median 0 n=1; mstar-pr: median 166 n=11; mstar-main: median 86 n=6
main_mstar-main_python_r1: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-main_python_r2: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-main_python_r2_retry: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-main_python_r3: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r1: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r1_retry: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r2: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r2_retry: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r3: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r3_retry: failed captures 0, graph runtime rust True, rust frontend False
```

</details>

<details><summary>profile phases_c8</summary>

```

=== worker_0  iters 100-100  records=1  batch=1.00 (range 1.0-1.0) ===
    work: 100 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    6.270    0.000   40.340       99 6207300.0 
engine.admit                                 0.010    0.010    0.020       89    8900.0 
engine.collect_outputs                       0.090    0.050    0.290      100   90000.0 
engine.commit                                0.060    0.040    0.120       89   53400.0 
engine.declare_step                          0.010    0.010    0.020      100   10000.0 
engine.forward                               6.300    0.100   40.250      100 6300000.0 
engine.plan.fresh                            1.120    1.350    1.900        6   67200.0 
engine.plan.promoted                         0.090    0.080    0.110       83   74700.0 
engine.postprocess                           0.010    0.000    0.040      100   10000.0 
engine.pre_admit                             0.020    0.020    0.020       83   16600.0 
engine.pre_declare                           0.010    0.010    0.020       83    8300.0 
engine.preprocess                            0.100    0.070    0.310      100  100000.0 
gpu_exec                                    12.170    6.250   40.530       9811926600.0 
iter_total                                  11.770    6.240   42.880      10011770000.0 
res.admit.semantic_sampler                   0.000    0.000    0.000      270       0.0*
res.admit.slow_ar_attn                       0.000    0.000    0.000      270       0.0*
res.admit.slow_ar_kv                         0.010    0.010    0.030      270   27000.0*
res.admit.slow_ar_pos                        0.000    0.000    0.000      270       0.0*
res.commit.sampler.offset                    0.030    0.020    0.090       89   26700.0 
res.commit.semantic_sampler                  0.040    0.020    0.090       89   35600.0 
res.commit.slow_ar_attn                      0.000    0.000    0.000       89       0.0 
res.commit.slow_ar_kv                        0.010    0.010    0.010       89    8900.0 
res.commit.slow_ar_pos                       0.010    0.010    0.010       89    8900.0 
res.plan.semantic_sampler                    0.290    0.070    0.310      353 1023700.0*
res.plan.slow_ar_attn                       41.110    0.280    3.500      353145118300.0*
res.plan.slow_ar_kv                          0.430    0.510    0.640      353 1517900.0*
res.plan.slow_ar_pos                         0.060    0.050    0.130      353  211800.0*
res.pre_plan.semantic_sampler                0.010    0.010    0.010       83    8300.0 
res.pre_plan.slow_ar_attn                    0.140    0.140    0.190       83  116200.0 
res.pre_plan.slow_ar_kv                      0.210    0.190    0.390       83  174300.0 
res.pre_plan.slow_ar_pos                     0.040    0.030    0.040       83   33200.0 
rust.send_outputs.call                       0.020    0.010    0.050       97   19400.0 
rust.send_outputs.marshal                    0.020    0.000    0.090       97   19400.0 
speculate                                    0.070    0.070    0.090       78   54600.0 
submit_spec                                  1.150    0.770    5.080       92 1058000.0 
worker.await_admit_settled                   0.720    0.400    2.930       14  100800.0 
worker.gpu_thread.await_plan                 0.240    0.260    0.360       92  220800.0 
worker.gpu_thread.exec                       6.790    0.460   40.510      100 6790000.0 
worker.gpu_thread.prepare_inputs             0.520    0.040    5.740      100  520000.0 
worker.plan_thread.await_commit              6.640    0.000   10.940       84 5577600.0 
worker.plan_thread.pre_plan                  0.460    0.430    0.690       83  381800.0 
worker.plan_thread.reserve_slot              0.020    0.020    0.030       84   16800.0 
worker.postprocess.check_stop                0.170    0.130    0.250       97  164900.0 
worker.postprocess.check_stop_tail           0.020    0.020    0.030       97   19400.0 
worker.postprocess.cleanup_inputs            0.010    0.010    0.010       99    9900.0 
worker.postprocess.completion_event_sync     4.080    4.870    5.020       97 3957600.0 
worker.postprocess.event_sync                4.100    4.870    5.570       99 4059000.0 
worker.postprocess.pending_loop_stops        0.010    0.010    0.010       97    9700.0 
worker.postprocess.prematerialize            0.160    0.120    0.240       97  155200.0 
worker.postprocess.register_outputs          0.020    0.000    0.190       97   19400.0 
worker.postprocess.route                     0.030    0.030    0.040       97   29100.0 
worker.postprocess.route_outputs             0.090    0.090    0.120       97   87300.0 
worker.postprocess.send                      0.040    0.010    0.140       97   38800.0 
worker.postprocess.send_outputs              0.090    0.060    0.200       97   87300.0 
worker.postprocess.send_prep                 0.010    0.010    0.010       97    9700.0 
worker.postprocess.stop_loops                0.000    0.000    0.000       97       0.0 
worker.postprocess.store_tensors             0.050    0.050    0.060       97   48500.0 
worker.postprocess.update_lru                0.000    0.000    0.000       97       0.0 
worker.postprocess_batch                     4.510    5.190    5.600       99 4464900.0 
worker.schedule                              0.020    0.010    0.080       74   14800.0 
worker.schedule_yield_away                   0.060    0.060    0.090       23   13800.0 
worker.submit_spec.launch_wait               1.130    0.750    5.070       92 1039600.0 
    * sampled on passes that ran no batch, so its time is spread over work it did not all do: us/1k
```

</details>

## Orpheus 3B 0.1 ft (voice tara, 24 kHz; M* single-GPU colocated config)

Phases completed: collect, gate, profile, rounds, select, setup, summarize, tune.

Tuning (score = geometric-mean audio-s/s over the tune points):

| engine | knob set | per-point audio-s/s | TTFA p50 c=1 | score | chosen |
|---|---|---|---:|---:|---|
| mstar-pr | python | c=1: 4.7, c=32: 64.9 | 0.083 | 17.39 |  |
| mstar-pr | rust | c=1: 4.7, c=32: 65.7 | 0.082 | 17.50 | yes |
| voxserve | bs64 | c=1: 2.4, c=32: 38.7 | 0.146 | 9.69 |  |
| voxserve | bs64_compile | c=1: 2.4, c=32: 38.9 | 0.146 | 9.72 | yes |
| voxserve | bs64_async | c=1: 2.4, c=32: 38.5 | 0.151 | 9.70 |  |
| orpheusvllm | default | c=1: 4.1, c=32: 63.6 | 0.087 | 16.09 |  |
| orpheusvllm | ignore_seed | c=1: 4.1, c=32: 64.9 | 0.087 | 16.24 | yes |

Arm configs: M* PR = rust; M* main = rust; vox-serve 0.1.0 = bs64_compile; vLLM 0.30.0 + SNAC (thin server) = ignore_seed

## audio-s/s: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 4.66 [4.66-4.66] n=3 | 7.78 [7.72-7.84] n=3 | 14.69 [14.63-14.72] n=3 | 25.29 [25.14-25.43] n=3 | 42.78 [42.18-43.45] n=3 | 64.72 [63.68-65.72] n=3 | 93.00 [92.69-93.36] n=3 |
| M* main | 4.41 [4.40-4.42] n=3 | 7.28 [7.19-7.33] n=3 | 13.94 [13.83-14.11] n=3 | 24.51 [24.01-24.86] n=3 | 40.03 [39.05-40.72] n=3 | 59.91 [59.05-60.72] n=3 | 80.90 [80.42-81.85] n=3 |
| vox-serve 0.1.0 | 2.43 [2.42-2.44] n=3 | 4.45 [4.42-4.47] n=3 | 8.40 [8.27-8.50] n=3 | 14.56 [14.53-14.62] n=3 | 24.17 [24.06-24.37] n=3 | 39.36 [39.13-39.73] n=3 | 58.13 [56.87-59.37] n=3 |
| vLLM 0.30.0 + SNAC (thin server) | 4.06 [4.06-4.07] n=3 | 7.39 [7.28-7.59] n=3 | 13.48 [13.42-13.55] n=3 | 24.28 [24.16-24.47] n=3 | 39.39 [38.86-39.83] n=3 | 66.67 [64.60-67.88] n=3 | 102.41 [98.78-104.30] n=3 |

## RTF p50: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.214 [0.214-0.214] n=3 | 0.249 [0.249-0.250] n=3 | 0.263 [0.263-0.264] n=3 | 0.295 [0.293-0.295] n=3 | 0.337 [0.336-0.338] n=3 | 0.444 [0.442-0.447] n=3 | 0.643 [0.641-0.646] n=3 |
| M* main | 0.226 [0.226-0.227] n=3 | 0.266 [0.266-0.266] n=3 | 0.279 [0.279-0.280] n=3 | 0.305 [0.302-0.307] n=3 | 0.366 [0.362-0.370] n=3 | 0.484 [0.481-0.488] n=3 | 0.741 [0.739-0.743] n=3 |
| vox-serve 0.1.0 | 0.412 [0.411-0.413] n=3 | 0.437 [0.436-0.439] n=3 | 0.463 [0.461-0.467] n=3 | 0.513 [0.511-0.516] n=3 | 0.585 [0.582-0.590] n=3 | 0.714 [0.713-0.714] n=3 | 0.982 [0.974-0.986] n=3 |
| vLLM 0.30.0 + SNAC (thin server) | 0.246 [0.245-0.246] n=3 | 0.258 [0.255-0.261] n=3 | 0.283 [0.280-0.287] n=3 | 0.305 [0.297-0.310] n=3 | 0.362 [0.353-0.379] n=3 | 0.420 [0.410-0.436] n=3 | 0.544 [0.542-0.545] n=3 |

## TTFA p50 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.082 [0.082-0.082] n=3 | 0.100 [0.100-0.100] n=3 | 0.105 [0.105-0.105] n=3 | 0.116 [0.116-0.116] n=3 | 0.131 [0.130-0.132] n=3 | 0.170 [0.169-0.171] n=3 | 0.245 [0.245-0.246] n=3 |
| M* main | 0.087 [0.086-0.087] n=3 | 0.106 [0.105-0.107] n=3 | 0.111 [0.110-0.112] n=3 | 0.120 [0.119-0.120] n=3 | 0.141 [0.140-0.142] n=3 | 0.185 [0.185-0.186] n=3 | 0.282 [0.279-0.284] n=3 |
| vox-serve 0.1.0 | 0.146 [0.146-0.147] n=3 | 0.156 [0.156-0.156] n=3 | 0.165 [0.165-0.166] n=3 | 0.180 [0.179-0.181] n=3 | 0.210 [0.210-0.210] n=3 | 0.255 [0.255-0.255] n=3 | 0.351 [0.349-0.352] n=3 |
| vLLM 0.30.0 + SNAC (thin server) | 0.087 [0.087-0.087] n=3 | 0.096 [0.096-0.097] n=3 | 0.108 [0.107-0.109] n=3 | 0.116 [0.113-0.118] n=3 | 0.137 [0.133-0.143] n=3 | 0.163 [0.161-0.168] n=3 | 0.212 [0.211-0.214] n=3 |

## TTFA p99 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.085 [0.083-0.088] n=3 | 0.102 [0.102-0.103] n=3 | 0.112 [0.110-0.113] n=3 | 0.125 [0.123-0.127] n=3 | 0.145 [0.142-0.147] n=3 | 0.193 [0.192-0.194] n=3 | 0.274 [0.272-0.277] n=3 |
| M* main | 0.088 [0.088-0.089] n=3 | 0.109 [0.109-0.110] n=3 | 0.118 [0.115-0.121] n=3 | 0.130 [0.129-0.134] n=3 | 0.154 [0.152-0.155] n=3 | 0.209 [0.206-0.212] n=3 | 0.315 [0.312-0.319] n=3 |
| vox-serve 0.1.0 | 0.148 [0.147-0.151] n=3 | 0.168 [0.165-0.171] n=3 | 0.200 [0.200-0.201] n=3 | 0.268 [0.267-0.269] n=3 | 0.416 [0.413-0.418] n=3 | 0.692 [0.690-0.694] n=3 | 1.252 [1.248-1.255] n=3 |
| vLLM 0.30.0 + SNAC (thin server) | 0.089 [0.088-0.090] n=3 | 0.099 [0.098-0.100] n=3 | 0.117 [0.116-0.119] n=3 | 0.122 [0.120-0.124] n=3 | 0.149 [0.146-0.152] n=3 | 0.178 [0.177-0.178] n=3 | 0.224 [0.223-0.227] n=3 |

## Gate (WER, Whisper large-v3)

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

Server commands (round 1):

```
/root/work/mstar-main/.venv/bin/python -m mstar.cli.main serve orpheus --config /root/work/mstar-main/configs/orpheus_colocated.yaml --gpus 0 --port 8400 --rust-frontend
/root/work/mstar/.venv/bin/python -m mstar.cli.main serve orpheus --config /root/work/mstar/configs/orpheus_colocated.yaml --gpus 0 --port 8400 --rust-frontend
/root/work/engines/orpheusvllm/bin/python /root/work/dbench/orpheus_vllm_server.py --host 127.0.0.1 --port 8400 --model canopylabs/orpheus-3b-0.1-ft --tokenizer canopylabs/orpheus-3b-0.1-ft --edges full --max-model-len 4096 --gpu-memory-utilization 0.85 --ignore-seed
/root/work/engines/voxserve/bin/vox-serve --model canopylabs/orpheus-3b-0.1-ft --host 127.0.0.1 --port 8400 --max-batch-size 64 --enable-torch-compile
```

<details><summary>notes</summary>

```
notes for orpheus
dropped engines: {}
startup s: voxserve: median 6 n=7; orpheusvllm: median 36 n=6; mstar-pr: median 42 n=9; mstar-main: median 211 n=4
main_mstar-main_rust_r1: failed captures 0, graph runtime rust True, rust frontend True
main_mstar-main_rust_r2: failed captures 0, graph runtime rust True, rust frontend True
main_mstar-main_rust_r3: failed captures 0, graph runtime rust True, rust frontend True
main_mstar-pr_rust_r1: failed captures 0, graph runtime rust True, rust frontend True
main_mstar-pr_rust_r2: failed captures 0, graph runtime rust True, rust frontend True
main_mstar-pr_rust_r3: failed captures 0, graph runtime rust True, rust frontend True
```

</details>

<details><summary>profile phases_c64</summary>

```

=== worker_0  iters 100-500  records=5  batch=51.55 (range 44.5-56.3) ===
    work: 25,774 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    8.869    0.000    0.920      499  171707.1 
engine.admit                                 0.022    0.010    0.110      397     332.9 
engine.collect_outputs                       0.234    0.130    0.820      500    4539.5 
engine.commit                                0.670    0.870    1.010      397   10315.0 
engine.declare_step                          0.134    0.200    0.330      500    2599.5 
engine.forward                               8.984    0.180    0.710      500  174284.2 
engine.plan.fresh                            4.228    1.905   25.070       11    1804.5 
engine.plan.promoted                         0.280    0.280    0.300      386    4193.4 
engine.postprocess                           0.032    0.020    0.160      500     620.8 
engine.pre_admit                             0.106    0.100    0.130      386    1584.2 
engine.pre_declare                           0.040    0.040    0.040      386     599.1 
engine.preprocess                            0.066    0.040    0.260      500    1280.4 
gpu_exec                                    13.095    4.150    7.150      499  253520.2 
iter_total                                  13.748    4.290   13.750      500  266702.9 
res.admit.attn                               0.000    0.000    0.000      251       0.0 
res.admit.kv_cache                           0.025    0.130    0.290      251     243.3 
res.admit.rope                               0.000    0.000    0.000      251       0.0 
res.admit.sampler                            0.000    0.000    0.000      251       0.0 
res.commit.attn                              0.000    0.000    0.000      397       0.0 
res.commit.kv_cache                          0.056    0.050    0.080      397     858.6 
res.commit.rope                              0.034    0.030    0.040      397     519.5 
res.commit.sampler                           0.574    0.780    0.910      397    8840.3 
res.commit.sampler.offset                    0.127    0.170    0.190      396    1957.8 
res.plan.attn                              121.076    0.000    3.670      637 2992363.6*
res.plan.kv_cache                            0.282    0.010    0.680      637    6961.7*
res.plan.rope                                0.032    0.000    0.130      637     802.4*
res.plan.sampler                             0.326    0.250    0.310      637    8067.8*
res.pre_plan.attn                            0.154    0.150    0.290      386    2307.0 
res.pre_plan.kv_cache                        0.315    0.270    0.730      386    4713.3 
res.pre_plan.rope                            0.078    0.070    0.090      386    1163.2 
res.pre_plan.sampler                         0.034    0.010    0.180      386     504.0 
rust.send_outputs.call                       0.070    0.050    0.180      499    1354.9 
rust.send_outputs.marshal                    0.056    0.000    0.390      499    1082.9 
speculate                                    0.269    0.260    0.320      326    3403.8 
submit_spec                                  1.416    1.550    1.920      499   27411.0 
worker.await_admit_settled                  12.360    9.730   25.260        3    1438.7 
worker.gpu_thread.await_plan                 0.286    0.330    0.560      499    5536.6 
worker.gpu_thread.exec                      10.440    1.840    2.520      500  202529.7 
worker.gpu_thread.prepare_inputs             0.218    0.120    0.600      500    4229.1 
worker.plan_thread.await_commit             11.090    0.000    0.040      396  170389.2 
worker.plan_thread.pre_plan                  0.758    0.660    1.360      386   11346.3 
worker.plan_thread.reserve_slot              0.020    0.020    0.030      396     307.3 
worker.postprocess.check_stop                0.138    0.080    0.610      499    2670.9 
worker.postprocess.check_stop_tail           0.094    0.100    0.210      499    1820.1 
worker.postprocess.cleanup_inputs            0.068    0.080    0.090      499    1316.4 
worker.postprocess.completion_event_sync     0.471    0.090    3.820      499    9110.0 
worker.postprocess.event_sync                0.461    0.080    3.810      499    8916.3 
worker.postprocess.pending_loop_stops        0.010    0.010    0.020      499     193.6 
worker.postprocess.prematerialize            0.074    0.000    0.600      499    1431.7 
worker.postprocess.register_outputs          0.108    0.080    0.280      499    2090.5 
worker.postprocess.route                     0.116    0.130    0.150      499    2246.1 
worker.postprocess.route_outputs             0.496    0.580    0.700      499    9603.1 
worker.postprocess.send                      0.130    0.060    0.560      499    2514.9 
worker.postprocess.send_outputs              0.918    1.010    1.120      499   17772.6 
worker.postprocess.send_prep                 0.096    0.030    0.170      499    1860.8 
worker.postprocess.stop_loops                0.000    0.000    0.050      499       0.0 
worker.postprocess.store_tensors             0.354    0.430    0.520      499    6853.8 
worker.postprocess.update_lru                0.010    0.010    0.010      499     193.6 
worker.postprocess_batch                     2.256    2.060    5.280      499   43686.3 
worker.schedule                              0.010    0.000    0.010       91      35.3 
worker.schedule_yield_away                   0.490    0.640    1.000      173    3287.8 
worker.submit_spec.launch_wait               1.398    1.530    1.910      499   27062.5 
    * sampled on passes that ran no batch, so its time is spread over work it did not all do: 
```

</details>

## Qwen3-TTS 0.6B CustomVoice (voice vivian, English, 24 kHz)

Phases completed: collect, gate, profile, rounds, select, setup, summarize, tune.

Tuning (score = geometric-mean audio-s/s over the tune points):

| engine | knob set | per-point audio-s/s | TTFA p50 c=1 | score | chosen |
|---|---|---|---:|---:|---|
| vllmomni | default | c=1: 14.1, c=32: 125.7 | 0.023 | 42.02 |  |
| vllmomni | high_conc | c=1: 14.0, c=32: 138.9 | 0.023 | 44.16 | yes |
| sgomni | base | c=1: 19.1, c=32: 264.0 | 0.012 | 71.07 | yes |
| sgomni | mrr128 | c=1: 19.1, c=32: 261.5 | 0.013 | 70.69 |  |

Arm configs: M* PR = python; M* main = python; vllm-omni 0.30.0 = high_conc; SGLang-Omni 0.1.9 = base

## audio-s/s: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 17.52 [17.36-17.66] n=3 | 24.28 [24.23-24.35] n=3 | 43.44 [43.26-43.61] n=3 | 69.67 [69.56-69.81] n=3 | 102.91 [102.40-103.67] n=3 | 146.15 [145.47-147.48] n=3 | 200.90 [198.16-202.80] n=3 |
| M* main | 16.14 [16.04-16.22] n=3 | 23.43 [23.36-23.49] n=3 | 41.30 [40.99-41.62] n=3 | 66.29 [66.09-66.64] n=3 | 98.87 [98.47-99.34] n=3 | 139.63 [139.41-139.80] n=3 | 148.44 [146.36-151.81] n=3 |
| vllm-omni 0.30.0 | 13.95 [13.90-13.98] n=3 | 24.57 [24.25-24.83] n=3 | 43.99 [43.79-44.35] n=3 | 69.30 [69.15-69.57] n=3 | 101.53 [101.01-101.88] n=3 | 138.56 [137.07-140.59] n=3 | 178.82 [177.86-179.46] n=3 |
| SGLang-Omni 0.1.9 | 19.16 [19.14-19.17] n=3 | 34.57 [34.36-34.81] n=3 | 66.59 [66.37-66.88] n=3 | 114.96 [114.30-115.38] n=3 | 182.21 [179.13-185.44] n=3 | 266.23 [263.35-268.51] n=3 | 360.47 [356.98-365.68] n=3 |

## RTF p50: mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.057 [0.057-0.058] n=3 | 0.081 [0.081-0.082] n=3 | 0.090 [0.089-0.090] n=3 | 0.108 [0.107-0.110] n=3 | 0.146 [0.144-0.148] n=3 | 0.211 [0.206-0.214] n=3 | 0.305 [0.304-0.307] n=3 |
| M* main | 0.062 [0.062-0.063] n=3 | 0.084 [0.083-0.084] n=3 | 0.093 [0.093-0.094] n=3 | 0.114 [0.112-0.114] n=3 | 0.151 [0.150-0.151] n=3 | 0.217 [0.215-0.220] n=3 | 0.432 [0.421-0.439] n=3 |
| vllm-omni 0.30.0 | 0.072 [0.071-0.072] n=3 | 0.079 [0.078-0.080] n=3 | 0.088 [0.087-0.089] n=3 | 0.108 [0.108-0.109] n=3 | 0.146 [0.145-0.146] n=3 | 0.213 [0.212-0.215] n=3 | 0.336 [0.330-0.341] n=3 |
| SGLang-Omni 0.1.9 | 0.052 [0.052-0.052] n=3 | 0.056 [0.055-0.056] n=3 | 0.058 [0.058-0.059] n=3 | 0.064 [0.064-0.065] n=3 | 0.080 [0.080-0.080] n=3 | 0.109 [0.108-0.110] n=3 | 0.168 [0.165-0.171] n=3 |

## TTFA p50 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.017 [0.017-0.018] n=3 | 0.027 [0.026-0.027] n=3 | 0.034 [0.033-0.035] n=3 | 0.039 [0.038-0.040] n=3 | 0.043 [0.040-0.047] n=3 | 0.052 [0.050-0.054] n=3 | 0.069 [0.067-0.071] n=3 |
| M* main | 0.018 [0.018-0.018] n=3 | 0.029 [0.028-0.029] n=3 | 0.036 [0.035-0.036] n=3 | 0.041 [0.041-0.042] n=3 | 0.051 [0.051-0.052] n=3 | 0.075 [0.071-0.077] n=3 | 0.080 [0.080-0.081] n=3 |
| vllm-omni 0.30.0 | 0.023 [0.023-0.023] n=3 | 0.033 [0.032-0.033] n=3 | 0.039 [0.037-0.041] n=3 | 0.046 [0.046-0.047] n=3 | 0.069 [0.066-0.072] n=3 | 0.101 [0.095-0.105] n=3 | 0.151 [0.148-0.155] n=3 |
| SGLang-Omni 0.1.9 | 0.012 [0.012-0.012] n=3 | 0.015 [0.015-0.016] n=3 | 0.016 [0.015-0.017] n=3 | 0.017 [0.017-0.017] n=3 | 0.019 [0.018-0.019] n=3 | 0.022 [0.022-0.023] n=3 | 0.027 [0.026-0.028] n=3 |

## TTFA p99 (s): mean [min-max] over rounds (n rounds); FAILED if any round failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M* PR | 0.022 [0.019-0.025] n=3 | 0.036 [0.036-0.037] n=3 | 0.047 [0.043-0.053] n=3 | 0.062 [0.058-0.064] n=3 | 0.079 [0.077-0.080] n=3 | 0.109 [0.102-0.122] n=3 | 0.170 [0.159-0.175] n=3 |
| M* main | 0.020 [0.019-0.022] n=3 | 0.038 [0.037-0.039] n=3 | 0.046 [0.044-0.049] n=3 | 0.060 [0.055-0.065] n=3 | 0.087 [0.081-0.098] n=3 | 0.128 [0.124-0.132] n=3 | 0.204 [0.168-0.271] n=3 |
| vllm-omni 0.30.0 | 0.029 [0.028-0.030] n=3 | 0.044 [0.041-0.046] n=3 | 0.054 [0.049-0.066] n=3 | 0.067 [0.060-0.072] n=3 | 0.111 [0.105-0.116] n=3 | 0.166 [0.150-0.188] n=3 | 0.284 [0.262-0.301] n=3 |
| SGLang-Omni 0.1.9 | 0.014 [0.013-0.015] n=3 | 0.018 [0.018-0.019] n=3 | 0.024 [0.023-0.024] n=3 | 0.033 [0.032-0.033] n=3 | 0.044 [0.043-0.045] n=3 | 0.074 [0.074-0.075] n=3 | 0.127 [0.126-0.129] n=3 |

## Gate (WER, Whisper large-v3)

Whisper large-v3, 128 LibriSpeech test-clean sentences per arm. Best WER 1.82%, same-code spread (two M* PR processes) 0.14 pt, tolerance 1.00 pt; median s/word over arms 0.393

Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within tolerance and speech rate within 25%) is annotated, not gating.

| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |
|---|---:|---:|---:|---:|---|---|
| mstar_main | 1.82 | 128 | 0 | 0.393 | PASS | yes |
| mstar_pr | 1.96 | 128 | 0 | 0.398 | PASS | yes |
| mstar_pr_b | 1.82 | 128 | 0 | 0.393 | PASS | yes |
| sgomni__base | 2.14 | 128 | 0 | 0.390 | PASS | yes |
| sgomni__mrr128 | 2.23 | 128 | 0 | 0.387 | PASS | yes |
| vllmomni__default | 2.28 | 128 | 0 | 0.393 | PASS | yes |
| vllmomni__high_conc | 1.96 | 128 | 0 | 0.390 | PASS | yes |

Server commands (round 1):

```
/root/work/mstar-main/.venv/bin/python -m mstar.cli.main serve qwen3_tts --config /root/work/mstar-main/configs/qwen3tts.yaml --gpus 0 --port 8400
/root/work/mstar/.venv/bin/python -m mstar.cli.main serve qwen3_tts --config /root/work/mstar/configs/qwen3tts.yaml --gpus 0 --port 8400
/root/work/engines/sgomni/bin/sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --host 127.0.0.1 --port 8400 --config /root/work/dbench/tts_configs/sgomni_qwen3tts_0p6b.yaml
/root/work/engines/vllmomni/bin/vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni --host 127.0.0.1 --port 8400 --deploy-config /root/work/engines/vllmomni/lib/python3.12/site-packages/vllm_omni/deploy/qwen3_tts_high_concurrency_mrv2_single_gpu.yaml
```

<details><summary>notes</summary>

```
notes for qwen3tts_0p6b
dropped engines: {}
startup s: vllmomni: median 84 n=6; sgomni: median 89 n=6; mstar-pr: median 126 n=8; mstar-main: median 178 n=4
main_mstar-main_python_r1: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-main_python_r2: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-main_python_r3: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r1: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r2: failed captures 0, graph runtime rust True, rust frontend False
main_mstar-pr_python_r3: failed captures 0, graph runtime rust True, rust frontend False
```

</details>

<details><summary>profile phases_c1</summary>

```

=== worker_0  iters 100-1600  records=16  batch=1.00 (range 1.0-1.0) ===
    work: 1,600 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    0.036    0.000    0.850     1599   35550.0 
engine.admit                                 0.011    0.010    0.020     1482    9818.8 
engine.collect_outputs                       0.107    0.090    0.400     1600  106887.5 
engine.commit                                0.094    0.090    0.130     1482   86806.3 
engine.declare_step                          0.019    0.020    0.030     1600   19375.0 
engine.forward                               0.137    0.110    1.840     1600  136875.0 
engine.plan.fresh                            2.451    1.500   24.750       22   33706.2 
engine.plan.promoted                         0.161    0.160    0.220     1460  147206.2 
engine.postprocess                           0.011    0.010    0.090     1600   10625.0 
engine.pre_admit                             0.020    0.020    0.030     1460   18250.0 
engine.pre_declare                           0.020    0.020    0.030     1460   18250.0 
engine.preprocess                            0.113    0.110    0.150     1600  112500.0 
gpu_exec                                     4.011    3.960    5.110     1598 4005800.0 
iter_total                                   3.897    3.960    5.640     1600 3896875.0 
res.admit.code_predictor                     0.000    0.000    0.000      358       0.0 
res.admit.talker_attn                        0.000    0.000    0.000      286       0.0 
res.admit.talker_kv                          0.011    0.030    0.090      286    2050.0 
res.admit.talker_pos                         0.000    0.000    0.000      286       0.0 
res.admit.talker_sampler                     0.000    0.000    0.000      286       0.0 
res.commit.code_predictor                    0.029    0.020    0.060     1482   27193.8 
res.commit.sampler.offset                    0.020    0.020    0.030     2964   37050.0*
res.commit.talker_attn                       0.000    0.000    0.000     1482       0.0 
res.commit.talker_kv                         0.010    0.010    0.010     1482    9262.5 
res.commit.talker_pos                        0.010    0.010    0.010     1482    9262.5 
res.commit.talker_sampler                    0.042    0.040    0.060     1482   38756.2 
res.plan.code_predictor                      0.081    0.060    0.290     1818   91712.5*
res.plan.talker_attn                         0.710    0.000    3.470     1746  774587.5*
res.plan.talker_kv                           0.090    0.010    0.620     1746   98168.8*
res.plan.talker_pos                          0.012    0.000    0.130     1746   13312.5*
res.plan.talker_sampler                      0.090    0.070    0.240     1746   97700.0*
res.pre_plan.code_predictor                  0.012    0.010    0.010     1460   11325.0 
res.pre_plan.talker_attn                     0.151    0.140    0.330     1460  137425.0 
res.pre_plan.talker_kv                       0.219    0.210    0.320     1460  199612.5 
res.pre_plan.talker_pos                      0.039    0.040    0.060     1460   35387.5 
res.pre_plan.talker_sampler                  0.010    0.010    0.010     1460    9125.0 
rust.send_outputs.call                       0.011    0.010    0.070     1583   11118.8 
rust.send_outputs.marshal                    0.008    0.000    0.080     1583    8018.8 
speculate                                    0.071    0.070    0.090     1380   61462.5 
submit_spec                                  1.073    0.870    3.830     1543 1034668.8 
worker.await_admit_settled                   2.079    0.580   34.530       63   81850.0 
worker.gpu_thread.await_plan                 0.281    0.265    0.570     1543  270700.0 
worker.gpu_thread.exec                       0.756    0.680    2.570     1600  755606.2 
worker.gpu_thread.prepare_inputs             0.241    0.040    3.510     1600  240625.0 
worker.plan_thread.await_commit              0.007    0.000    0.010     1460    6231.3 
worker.plan_thread.pre_plan                  0.506    0.485    0.830     1460  461281.2 
worker.plan_thread.reserve_slot              0.027    0.025    0.040     1460   24525.0 
worker.postprocess.check_stop                0.040    0.030    0.150     1583   39537.5 
worker.postprocess.check_stop_tail           0.038    0.030    0.090     1583   37718.8 
worker.postprocess.cleanup_inputs            0.012    0.010    0.020     1599   11868.8 
worker.postprocess.completion_event_sync     2.309    2.510    4.000     1583 2284925.0 
worker.postprocess.event_sync                2.313    2.505    4.000     1599 2311087.5 
worker.postprocess.pending_loop_stops        0.010    0.010    0.010     1583    9893.8 
worker.postprocess.prematerialize            0.014    0.000    0.140     1583   13575.0 
worker.postprocess.register_outputs          0.013    0.000    0.150     1583   12968.8 
worker.postprocess.route                     0.027    0.020    0.040     1583   26581.2 
worker.postprocess.route_outputs             0.087    0.080    0.130     1583   86562.5 
worker.postprocess.send                      0.023    0.010    0.150     1583   22856.2 
worker.postprocess.send_outputs              0.080    0.070    0.730     1583   79100.0 
worker.postprocess.send_prep                 0.010    0.010    0.010     1583    9893.8 
worker.postprocess.stop_loops                0.000    0.000    0.000     1583       0.0 
worker.postprocess.store_tensors             0.052    0.050    0.080     1583   51937.5 
worker.postprocess.update_lru                0.000    0.000    0.000     1583       0.0 
worker.postprocess_batch                     2.577    2.725    4.470     1599 2575243.8 
worker.
```

</details>

<details><summary>profile phases_c32</summary>

```

=== worker_0  iters 100-100  records=1  batch=22.45 (range 22.4-22.4) ===
    work: 2,245 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    0.290    0.000    2.030       99   12788.4 
engine.admit                                 0.020    0.020    0.070       76     677.1 
engine.collect_outputs                       0.330    0.180    1.420      100   14699.3 
engine.commit                                0.240    0.180    0.690       76    8124.7 
engine.declare_step                          0.020    0.030    0.040      100     890.9 
engine.forward                               0.690    0.160    2.880      100   30735.0 
engine.plan.fresh                            3.780    1.550   24.800       11   18521.2 
engine.plan.promoted                         0.260    0.250    0.330       65    7527.8 
engine.postprocess                           0.070    0.050    0.210      100    3118.0 
engine.pre_admit                             0.070    0.060    0.090       65    2026.7 
engine.pre_declare                           0.040    0.030    0.050       65    1158.1 
engine.preprocess                            0.190    0.160    0.340      100    8463.3 
gpu_exec                                     7.990    6.190   19.040       98  348784.0 
iter_total                                   9.320    6.270   32.200      100  415144.8 
res.admit.code_predictor                     0.000    0.000    0.000      347       0.0*
res.admit.talker_attn                        0.000    0.000    0.000      275       0.0*
res.admit.talker_kv                          0.010    0.010    0.040      275    1224.9*
res.admit.talker_pos                         0.000    0.000    0.000      275       0.0*
res.admit.talker_sampler                     0.000    0.000    0.000      275       0.0*
res.commit.code_predictor                    0.060    0.030    0.150       76    2031.2 
res.commit.sampler.offset                    0.040    0.020    0.100      152    2708.2*
res.commit.talker_attn                       0.000    0.000    0.000       76       0.0 
res.commit.talker_kv                         0.030    0.030    0.050       76    1015.6 
res.commit.talker_pos                        0.020    0.020    0.030       76     677.1 
res.commit.talker_sampler                    0.130    0.100    0.310       76    4400.9 
res.plan.code_predictor                      0.120    0.070    0.300      412   22022.3*
res.plan.talker_attn                        42.750    0.280    3.540      340 6474387.5*
res.plan.talker_kv                           0.400    0.520    0.620      340   60579.1*
res.plan.talker_pos                          0.060    0.050    0.130      340    9086.9*
res.plan.talker_sampler                      0.170    0.050    0.250      340   25746.1*
res.pre_plan.code_predictor                  0.130    0.010    0.670       65    3763.9 
res.pre_plan.talker_attn                     0.190    0.170    0.370       65    5501.1 
res.pre_plan.talker_kv                       0.270    0.280    0.440       65    7817.4 
res.pre_plan.talker_pos                      0.060    0.050    0.100       65    1737.2 
res.pre_plan.talker_sampler                  0.070    0.010    0.230       65    2026.7 
rust.send_outputs.call                       0.060    0.040    0.190       99    2645.9 
rust.send_outputs.marshal                    0.070    0.000    0.310       99    3086.9 
speculate                                    0.200    0.170    0.310       49    4365.3 
submit_spec                                  2.350    1.690    5.100       99  103630.3 
worker.await_admit_settled                   4.470    1.020   35.420       12   23893.1 
worker.gpu_thread.await_plan                 0.270    0.220    0.940       99   11906.5 
worker.gpu_thread.exec                       2.280    1.180    6.240      100  101559.0 
worker.gpu_thread.prepare_inputs             1.670    0.220    7.510      100   74387.5 
worker.plan_thread.await_commit              0.200    0.000    1.180       75    6681.5 
worker.plan_thread.pre_plan                  0.880    0.660    1.680       65   25478.8 
worker.plan_thread.reserve_slot              0.020    0.020    0.040       75     668.2 
worker.postprocess.check_stop                0.210    0.080    0.970       99    9260.6 
worker.postprocess.check_stop_tail           0.080    0.100    0.150       99    3527.8 
worker.postprocess.cleanup_inputs            0.040    0.040    0.060       99    1763.9 
worker.postprocess.completion_event_sync     3.070    2.230   11.950       99  135380.8 
worker.postprocess.event_sync                3.060    2.220   11.940       99  134939.9 
worker.postprocess.pending_loop_stops        0.010    0.010    0.020       99     441.0 
worker.postprocess.prematerialize            0.150    0.000    0.960       99    6614.7 
worker.postprocess.register_outputs          0.080    0.000    0.390       99    3527.8 
worker.postprocess.route                     0.080    0.090    0.110       99    3527.8 
worker.postprocess.route_outputs             0.250    0.290    0.340       99   11024.5 
worker.postprocess.send                      0.140    0.060    0.500       99    6173.7 
worker.postprocess.send_outputs              0.660    0.430    1.780       99   29104.7 
worker.postprocess.send_prep                 0.030    0.020    0.100       99    1322.9 
worker.postprocess.stop_loops                0.010    0.000    0.060       99     441.0 
worker.postprocess.store_tensors             0.150    0.180    0.210       99    6614.7 
worker.postprocess.update_lru                0.000    0.010    0.010       99       0.0 
worker.postprocess_batch                     4.350    3.440   13.160       99  191826.3 
worker
```

</details>

## Not covered

- Qwen3-TTS: the official qwen-tts package has no HTTP server (correctness reference only, not timed); vox-serve was not run for Qwen3-TTS (its client sends no speaker, unverified on 12 Hz CustomVoice).
- Orpheus: vllm-omni and SGLang-Omni do not serve it; the official orpheus-speech package is a library (no server) pinned to vLLM 0.7.3, not run.
- fish-speech's S2-Pro support is only in prereleases (v3.0.0-beta); it serves one request at a time.
- Speaker similarity / MOS not measured; the gate is WER plus speech-rate and silence checks.

