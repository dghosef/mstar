# Diffusion lane: M* vs vllm-omni, SGLang diffusion and diffusers (node2, 8x H200)

_Generated 2026-10-23 14:34 UTC by report_diffusion.py from /workspace/results/bench/<model>/ on node2._

## Setup common to every model

**Hardware.** node2: dedicated AWS p5en.48xlarge via Eras Cloud, 8x H200 141 GB (NVLink), driver 580.178.04,
2x Xeon Platinum 8488C (192 vCPUs), 2 TiB RAM. Every timed run used GPU 0 with no other server on the box;
untimed correctness collection ran arms in parallel on separate GPUs.

**Arms and versions (pinned).**

| arm | version | torch | notes |
|---|---|---|---|
| M* PR | perf/overnight-all d248477ec | 2.12.1+cu132 | each model's shipped config (`configs/<model>.yaml`), Python API frontend, Rust graph runtime (`graph runtime: rust` in the worker log), `MSTAR_SHM_ARENA=1`, `MSTAR_REQUIRE_ACCELERATOR_GRAPHS=1` for the captured image models |
| M* main | main abdfd5802 | 2.12.1+cu132 | same settings, its own configs |
| vllm-omni | vllm-omni 0.30.0 on vllm 0.30.0 | 2.13.0+cu132 | `vllm serve <hf> --omni` + tuned knobs |
| SGLang diffusion | sglang 0.5.22 (`sglang[diffusion]`) | 2.14.1+cu132 | `sglang serve --model-path <hf> --num-gpus 1` + tuned knobs |
| diffusers | 0.41.0 | 2.14.1+cu132 | in-process pipeline; eager = the correctness reference; tuned = the fastest passing knob set among eager, regional (repeated-block) or whole-transformer torch.compile with dynamic shapes, x SDPA / cuDNN / FlashAttention-3 (hub kernel), +- compiled VAE decode |

M* frontend: the Rust API frontend does not serve these routes yet (`DiffusionImageAdapter` images and
`Wan22Adapter` videos are listed in `NOT_PORTED` in test/rust/test_rust_openai_adapters.py on both trees), so
both M* arms use the default Python frontend for every model.

**Matched settings.** bf16 weights everywhere; same resolution, steps, guidance, scheduler (each checkpoint's own),
prompts and seeds on every arm; noise from a CPU generator seeded per request (mstar's convention and its oracle
recorders'; vllm-omni and SGLang take `generator_device: cpu` per image request, SGLang also per video request).
No step caching (TeaCache / cache-dit / step cache off on every arm), no quantization, no reduced steps.

**Prompts.** Images: a fixed 64-prompt slice of PartiPrompts (P2, `nateraw/parti-prompts`, every 25th prompt,
indices 0, 25, ..., 1575; all 12 categories). Video: 16 VBench prompts (`prompts/all_dimension.txt`, every 59th
line). None of these were used in the overnight tuning.

**Client.** One Python client and policy for every served arm (`dclient.py`): closed loop with c workers;
request i gets prompt i mod N and seed base + i; each response is content-checked (decodes, right size / frame
count, not flat) and a failed check counts as a failed request. diffusers is not a server: its rows are
in-process offline batches of b = c (pipeline call only: no HTTP, no PNG/mp4 encode), labelled as such.

**Protocol.** Competitors tuned first (one server per knob set on GPU 0 alone, B=1 and c=8); knob sets whose
outputs fail the correctness gate are not eligible. Then 3 rounds; each round restarts every server (fresh
process) and rotates the arm order. Per server: a smoke request, one untimed pass over the whole prompt set
(compile / capture / per-shape warmup), then per point one warmup wave of c requests and the measured requests.
A point counts only with failed == 0 in every round. Mean and min-max over rounds; 1-min load average per cell.

**Correctness gate** (criteria fixed before any output was collected, in `dgate.py`): each arm's outputs vs
the diffusers eager reference on the same prompts and seeds; PSNR, SSIM and CLIP score (ViT-L/14). Yardstick Y =
the worst of the reference vs itself in a second process (ref_b), under memory-efficient SDPA (ref_k), and
under the tuned arm's torch.compile (ref_c). A seed-change control (ref_seed) shows what different noise looks
like. A seed-matched arm passes if median PSNR >= Y - 2 dB, p10 PSNR >= Y.p10 - 3 dB, median SSIM >= Y - 0.03,
mean CLIP >= reference - 1.0 and every output is valid. An arm whose noise is drawn differently gets a
distribution-level check only (CLIP and valid outputs), labelled as such.

**Gate amendment (2026-10-22 06:50 UTC, recorded in decisions.jsonl).** The gate was amended after observing that
independent engines are not pixel-matched to the diffusers reference (Z-Image: vllm-omni and SGLang at 23.8-24.3 dB
median PSNR, against a different-seed control of 10.9 dB, with equal CLIP). M* arms keep the original parity
criteria above. Competitor arms are checked for "the same image for the same seed": valid outputs, mean CLIP >=
reference - 0.5, and median PSNR >= (different-seed control median + yardstick median) / 2. Both verdicts are shown in
each gate table. For TTS, M* is held to the reference-parity tier and competitors to WER within tolerance.

**Gate rule as applied (2026-10-22 ~07:00 UTC, user direction, recorded in decisions.jsonl): sanity gate; parity
reported, not gating.** An arm (M* included) is excluded only if its output is clearly broken: invalid/blank
outputs, mean CLIP more than 2 points below the reference, or median PSNR within 3 dB of the different-seed control
(a different image); for TTS, silent/missing audio or WER > 2x the best engine's and > 10%. Strict parity (the
pre-registered criteria for M*, the amended same-image check for competitors, WER tolerance for TTS) is reported in
each gate table as "parity: yes / no (annotated)", next to the original pre-registered verdict.

## Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo), text to image

**Workload.** 1024x1024, 8 steps (8 DiT forwards), guidance 0 (Turbo, no CFG), FlowMatch Euler (checkpoint scheduler), bf16, 64 PartiPrompts.

Queue phases completed: collect, gallery, gate, knobs, profile, ref_c, rounds, select, setup, summarize, tune.

### Correctness gate (vs diffusers eager reference, same prompts and seeds)

Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR 27.94 dB, p10 PSNR 21.41 dB, median SSIM 0.9523; ref_a CLIP 27.66; different-seed control median PSNR 10.94 dB; competitor same-image threshold 19.44 dB (amended gate, see dgate.py)

| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | tier | verdict | original verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| ref_b | 31.10 | 23.29 | 18.36 | 0.9672 | 27.61 | 0 | yardstick | - | - |
| ref_k | 28.89 | 23.84 | 19.62 | 0.9586 | 27.57 | 0 | yardstick | - | - |
| ref_c | 27.94 | 21.41 | 15.24 | 0.9523 | 27.66 | 0 | yardstick | - | - |
| ref_seed | 10.94 | 6.79 | 4.64 | 0.4235 | 27.84 | 0 | yardstick | - | - |
| mstar_pr | 29.47 | 24.20 | 17.77 | 0.9655 | 27.58 | 0 | parity (M*) | PASS | PASS |
| mstar_pr_b | 30.19 | 23.49 | 15.36 | 0.9612 | 27.69 | 0 | parity (M*) | PASS | PASS |
| mstar_main | 29.95 | 24.85 | 15.53 | 0.9623 | 27.82 | 0 | parity (M*) | PASS | PASS |
| sglang__base | 24.26 | 20.09 | 14.74 | 0.8968 | 27.98 | 0 | same-image (competitor, amended) | PASS | FAIL |
| sglang__bcg | 24.11 | 20.09 | 14.74 | 0.8960 | 27.97 | 0 | same-image (competitor, amended) | PASS | FAIL |
| sglang__bcg_b8 | 24.11 | 20.09 | 14.74 | 0.8960 | 27.97 | 0 | same-image (competitor, amended) | PASS | FAIL |
| sglang__compile | 24.31 | 20.70 | 17.67 | 0.8896 | 27.92 | 0 | same-image (competitor, amended) | PASS | FAIL |
| sglang__compile_b8 | 24.30 | 20.15 | 17.67 | 0.8924 | 27.88 | 0 | same-image (competitor, amended) | PASS | FAIL |
| vllmomni__base | 23.81 | 18.79 | 13.66 | 0.9028 | 27.96 | 0 | same-image (competitor, amended) | PASS | FAIL |
| vllmomni__cudnn | 24.01 | 18.67 | 13.97 | 0.8984 | 27.93 | 0 | same-image (competitor, amended) | PASS | FAIL |
| vllmomni__eager | 23.81 | 18.87 | 13.94 | 0.9014 | 27.83 | 0 | same-image (competitor, amended) | PASS | FAIL |
| vllmomni__full | 23.75 | 18.91 | 13.54 | 0.9034 | 28.01 | 0 | same-image (competitor, amended) | PASS | FAIL |
| ref_a | - | - | - | - | 27.66 | 0 | reference | - | - |


Tags: ref_b = reference in a second process; ref_k = memory-efficient SDPA; ref_c = diffusers with the tuned arm's compile/attention; ref_seed = reference with other seeds (different noise); mstar_pr_b = a second M* PR server process (same-code control); `<engine>__<knobs>` = each competitor knob set from the tune pass. Contact sheet: `sheet_zimage.jpg` in the results folder.

### Competitor tuning (one server process per knob set, GPU 0 alone; score = geometric mean throughput over the tune points, img/s)

| engine | knob set | flags | per-point throughput | B=1 latency (s) | score | gate | chosen |
|---|---|---|---|---:|---:|---|---|
| vllm-omni | base | `(defaults)` | c=1: 1.045, c=8: 1.156 | 0.932 | 1.099 | PASS | yes |
| vllm-omni | cudnn | `--diffusion-attention-backend CUDNN_ATTN` | c=1: 1.022, c=8: 1.132 | 0.954 | 1.075 | PASS |  |
| vllm-omni | full | `--diffusion-compile-granularity full` | c=1: 1.040, c=8: 1.152 | 0.936 | 1.095 | PASS |  |
| vllm-omni | eager | `--enforce-eager` | c=1: 0.942, c=8: 1.031 | 1.036 | 0.986 | PASS |  |
| SGLang | base | `--performance-mode speed` | c=1: 1.261, c=8: 1.269 | 0.781 | 1.265 | PASS |  |
| SGLang | compile | `--performance-mode speed --enable-torch-compile true` | c=1: 1.205, c=8: 1.248 | 0.818 | 1.226 | PASS |  |
| SGLang | bcg | `--performance-mode speed --enable-breakable-cuda-graph true --warmup-resolutions 1024x1024` | c=1: 1.266, c=8: 1.276 | 0.778 | 1.271 | PASS | yes |
| SGLang | compile_b8 | `--performance-mode speed --enable-torch-compile true --batching-max-size 8` | c=1: 1.217, c=8: 0.333 | 0.810 | 0.637 | PASS |  |
| SGLang | bcg_b8 | `--performance-mode speed --enable-breakable-cuda-graph true --warmup-resolutions 1024x1024 --batching-max-size 8` | c=1: 1.238, c=8: 1.298 | 0.796 | 1.267 | PASS |  |
| diffusers | eager | `--compile none` | c=1: 0.777, c=8: 0.804 | 1.287 | 0.790 | n/a (reference impl.) |  |
| diffusers | regional_dyn | `--compile regional --dynamic` | c=1: 1.131, c=8: 0.990 | 0.884 | 1.058 | n/a (reference impl.) | yes |
| diffusers | regional_dyn_cudnn | `--compile regional --dynamic --attn _native_cudnn` | c=1: 1.117, c=8: 0.964 | 0.895 | 1.037 | n/a (reference impl.) |  |
| diffusers | regional_dyn_fa3 | `--compile regional --dynamic --attn _flash_3_hub` | failed | - | failed | n/a (reference impl.) |  |
| diffusers | regional_dyn_fa3_vae | `--compile regional --dynamic --attn _flash_3_hub --compile-vae` | failed | - | failed | n/a (reference impl.) |  |

### Exact server commands (round 1)

```
/root/work/mstar-main/.venv/bin/python -m mstar.cli.main serve z_image_turbo --config /root/work/mstar-main/configs/z_image_turbo.yaml --gpus 0 --port 8400
/root/work/mstar/.venv/bin/python -m mstar.cli.main serve z_image_turbo --config /root/work/mstar/configs/z_image_turbo.yaml --gpus 0 --port 8400
/root/work/engines/sgld/bin/sglang serve --model-path Tongyi-MAI/Z-Image-Turbo --num-gpus 1 --host 127.0.0.1 --port 8400 --scheduler-port 8500 --performance-mode speed --enable-breakable-cuda-graph true --warmup-resolutions 1024x1024
/root/work/engines/vllmomni/bin/vllm serve Tongyi-MAI/Z-Image-Turbo --omni --host 127.0.0.1 --port 8400
```

Versions: diffusers: {"packages": ["diffusers==0.41.0", "kernels==0.17.2", "torch==2.14.1+cu132", "transformers==5.20.2", "triton==3.8.0"]}; mstar-main: {"imports": "/root/work/mstar-main/mstar/__init__.py 2.12.1+cu132 0.41.0", "commit": "HEAD abdfd580296a843f62ce967ddb93b0a3dfb225e3"}; mstar-pr: {"imports": "/root/work/mstar/mstar/__init__.py 2.12.1+cu132 0.41.0", "commit": "perf/overnight-all d248477ec795c6ddf0716e23932be15be2b573e2"}; sglang: {"packages": ["cache-dit==1.5.1", "diffusers==0.37.0", "flash-attn-4==4.0.0b36", "flashinfer-python==0.7.1", "kernels==0.14.1", "kernels-data==0.16.2", "sgl-deep-ep==0.1.2.post1", "sgl-deep-gemm==0.2.2", "sglang==0.5.22", "sglang-kernel==0.4.9", "torch==2.14.1+cu132", "transformers==5.19.0", "triton; vllmomni: {"packages": ["cache-dit==1.5.0", "diffusers==0.40.0", "fa3-fwd==0.0.3", "flashinfer-python==0.6.18.post1", "kernels==0.16.1", "kernels-data==0.16.2", "torch==2.13.0+cu132", "transformers==5.14.1", "triton==3.7.1", "vllm==0.30.0", "vllm-omni==0.30.0"]}

### Timed results (3 rounds, fresh server per round, rotated order)

Arm configs: M* PR = config; M* main = config; vllm-omni = base; SGLang = bcg; diffusers (tuned, offline batch) = regional_dyn

## Throughput (img/s), steps=8: mean [min-max] over rounds; n rounds; failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* PR | 1.142 [1.140-1.144] n=3 | 1.192 [1.192-1.194] n=3 | 1.208 [1.206-1.209] n=3 | 1.214 [1.213-1.215] n=3 | 1.206 [1.205-1.208] n=3 |
| M* main | 1.123 [1.121-1.126] n=3 | 1.173 [1.170-1.176] n=3 | 1.104 [1.035-1.195] n=3 | 1.072 [1.040-1.105] n=3 | 1.174 [1.173-1.174] n=3 |
| vllm-omni | 1.035 [1.034-1.036] n=3 | 1.145 [1.144-1.146] n=3 | 1.149 [1.148-1.150] n=3 | 1.150 [1.150-1.151] n=3 | 1.149 [1.148-1.150] n=3 |
| SGLang | 1.260 [1.260-1.260] n=3 | 1.271 [1.271-1.271] n=3 | 1.274 [1.274-1.274] n=3 | 1.273 [1.273-1.274] n=3 | 1.271 [1.270-1.271] n=3 |
| diffusers (tuned, offline batch) | 1.120 [1.120-1.120] n=3 | 1.024 [1.023-1.026] n=3 | 1.098 [1.097-1.099] n=3 | 1.101 [1.100-1.102] n=3 | 1.081 [1.081-1.082] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* main | 1.02x | 1.02x | 1.09x | 1.13x | 1.03x |
| vllm-omni | 1.10x | 1.04x | 1.05x | 1.06x | 1.05x |
| SGLang | 0.91x | 0.94x | 0.95x | 0.95x | 0.95x |
| diffusers (tuned, offline batch) | 1.02x | 1.16x | 1.10x | 1.10x | 1.12x |

## B=1 latency (s), steps=8, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 0.846 [0.844-0.847] | 0.843 | 0.884 | 32 | 3 | 0 |
| M* main | 0.860 [0.858-0.862] | 0.858 | 0.894 | 32 | 3 | 0 |
| vllm-omni | 0.941 [0.940-0.943] | 0.940 | 0.972 | 32 | 3 | 0 |
| SGLang | 0.782 [0.782-0.783] | 0.781 | 0.805 | 32 | 3 | 0 |
| diffusers (tuned, offline batch) | 0.893 [0.892-0.893] | 0.887 | 0.997 | 32 | 3 | 0 |

1-min load average at cell start: median 1.9, max 19.4 over 75 cells (192 vCPUs).

### Profile (phases_c1, MSTAR_PHASE_TIMING=100, benchmark/worker_phases): GPU-bound (worker waits on the GPU most of each step); event_sync / iter_total up to 103%

<details><summary>phase tables</summary>

```
=== worker_0  iters 100-300  records=3  batch=1.00 (range 1.0-1.0) ===
    work: 300 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    0.533    0.000    4.210      299  530766.7 
engine.collect_outputs                       0.040    0.030    0.050      273   36400.0 
engine.declare_step                          0.000    0.000    0.000      273       0.0 
engine.forward                               0.988    0.530    4.250      273  899500.0 
engine.postprocess                           0.000    0.000    0.000      300       0.0 
engine.preprocess                            0.097    0.100    0.140      273   87966.7 
gpu_exec                                    72.077   93.770   97.530      29871596533.3 
iter_total                                  68.510   93.830   97.540      30068510000.0 
rust.send_outputs.call                       0.020    0.010    0.050      272   18133.3 
rust.send_outputs.marshal                    0.020    0.000    0.060      272   18133.3 
speculate                                    0.080    0.080    0.120      217   57866.7 
submit_spec                                  0.473    0.480    0.760      244  384966.7 
worker.await_admit_settled                   1.423    1.790   14.880       56  265700.0 
worker.gpu_thread.await_plan                 0.037    0.000    0.050      244   29766.7 
worker.gpu_thread.exec                       1.125    0.760    4.410      300 1125366.7 
worker.gpu_thread.prepare_inputs             0.293    0.130    1.560      300  292933.3 
worker.postprocess.check_stop                0.153    0.130    0.240      272  138966.7 
worker.postprocess.check_stop_tail           0.010    0.010    0.030      272    9066.7 
worker.postprocess.cleanup_inputs            0.010    0.010    0.020      299    9966.7 
worker.postprocess.completion_event_sync    77.844   92.860   96.810      27270579000.0 
worker.postprocess.event_sync               70.808   92.750   96.800      29970571766.7 
worker.postprocess.pending_loop_stops        0.010    0.010    0.010      272    9066.7 
worker.postprocess.prematerialize            0.143    0.120    0.230      272  129900.0 
worker.postprocess.register_outputs          0.070    0.000    0.280      272   63200.0 
worker.postprocess.route                     0.030    0.030    0.040      272   27200.0 
worker.postprocess.route_outputs             0.097    0.090    0.150      272   87666.7 
worker.postprocess.send                      0.040    0.010    0.120      272   36266.7 
worker.postprocess.send_outputs              0.050    0.030    0.140      272   45333.3 
worker.postprocess.send_prep                 0.010    0.010    0.010      272    9066.7 
worker.postprocess.stop_loops                0.000    0.000    0.020      272       0.0 
worker.postprocess.store_tensors             0.050    0.050    0.080      272   45333.3 
worker.postprocess.update_lru                0.000    0.000    0.000      272       0.0 
worker.postprocess_batch                    71.181   93.030   97.060      29970943566.7 
worker.schedule                              0.013    0.010    0.090      734   32933.3*
worker.schedule_yield_away                   0.030    0.010    0.090       82    8200.0 
worker.submit_spec.launch_wait               0.450    0.460    0.750      244  365966.7 
    * sampled on passes that ran no batch, so its time is spread over work it did not all do: us/1k is an upper bound for this row
```

</details>

### Summary (generated from the tables above)

- vs M* main: throughput 1.02x (c=1) to 1.13x (c=8); B=1 latency 0.846 s vs 0.860 s (0.98x). M* PR is not slower at any point.
- vs vllm-omni: throughput 1.04x (c=2) to 1.10x (c=1); B=1 latency 0.846 s vs 0.941 s (0.90x). M* PR is not slower at any point.
- vs SGLang: throughput 0.91x (c=1) to 0.95x (c=8); B=1 latency 0.846 s vs 0.782 s (1.08x). M* PR is slower at c=1 (0.91x), c=2 (0.94x), c=4 (0.95x), c=8 (0.95x), c=16 (0.95x).
- vs diffusers (offline batch, no serving): throughput 1.02x (c=1) to 1.16x (c=2); B=1 latency 0.846 s vs 0.893 s (0.95x). M* PR is not slower at any point.

<details><summary>notes.txt (startup times, health checks, decisions)</summary>

```
notes for zimage (2026-10-22 08:15 UTC)

server startup seconds per arm (all starts, incl. cold compiles): vllmomni: min 21 median 24 max 36 n=17; sglang: min 36 median 36 max 132 n=11; mstar-main: min 111 median 114 max 1299 n=6; mstar-pr: min 214 median 214 max 1705 n=7

health (per server log):
  main_mstar-main_config_r1: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-main_config_r2: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-main_config_r3: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r1: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r2: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r3: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_sglang_bcg_r1: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_sglang_bcg_r2: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_sglang_bcg_r3: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_vllmomni_base_r1: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  main_vllmomni_base_r2: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  main_vllmomni_base_r3: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_base_sglang_base_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_base_vllmomni_base_r0: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_base_vllmomni_base_r0_retry: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_bcg_b8_sglang_bcg_b8_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_bcg_sglang_bcg_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_compile_b8_sglang_compile_b8_r0: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-22 05:27:33] Configuring layerwise offload for ZImageTransformer2DModel: layers (3"], "errors": 30}
  tune_compile_b8_sglang_compile_b8_r0_retry: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-22 05:43:31] Configuring layerwise offload for ZImageTransformer2DModel: layers (3"], "errors": 0}
  tune_compile_sglang_compile_r0: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-22 05:20:57] Configuring layerwise offload for ZImageTransformer2DModel: layers (3"], "errors": 0}
  tune_cudnn_vllmomni_cudnn_r0: {"attention": ["CUDNN_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_cudnn_vllmomni_cudnn_r0_retry: {"attention": ["CUDNN_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_eager_vllmomni_eager_r0: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  tune_full_vllmomni_full_r0: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  tune_full_vllmomni_full_r0_retry: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}

decisions:
  2026-10-22 04:50:17 retry: {"tag": "tune_base", "arm": "vllmomni", "knob": "base", "round": 0}
  2026-10-22 04:50:40 pass_failed: {"tag": "tune_base", "arm": "vllmomni", "knob": "base", "round": 0}
  2026-10-22 04:51:09 retry: {"tag": "tune_cudnn", "arm": "vllmomni", "knob": "cudnn", "round": 0}
  2026-10-22 04:51:35 pass_failed: {"tag": "tune_cudnn", "arm": "vllmomni", "knob": "cudnn", "round": 0}
  2026-10-22 04:52:09 retry: {"tag": "tune_full", "arm": "vllmomni", "knob": "full", "round": 0}
  2026-10-22 04:52:38 pass_failed: {"tag": "tune_full", "arm": "vllmomni", "knob": "full", "round": 0}
  2026-10-22 04:59:06 retry: {"tag": "tune_base", "arm": "vllmomni", "knob": "base", "round": 0}
  2026-10-22 05:00:26 pass_failed: {"tag": "tune_base", "arm": "vllmomni", "knob": "base", "round": 0}
  2026-10-22 05:43:00 retry: {"tag": "tune_compile_b8", "arm": "sglang", "knob": "compile_b8", "round": 0}
  2026-10-22 05:53:02 phase_error: {"phase": "tune", "crashes": 1, "error": "FileNotFoundError(2, 'No such file or directory')"}
  2026-10-22 05:53:33 phase_error: {"phase": "tune", "crashes": 2, "error": "FileNotFoundError(2, 'No such file or directory')"}
  2026-10-22 05:57:09 retry: {"tag": "tune_regional", "arm": "diffusers", "knob": "regional", "round": 0}
  2026-10-22 05:58:04 retry: {"tag": "tune_full", "arm": "diffusers", "knob": "full", "round": 0}
  2026-10-22 05:58:29 retry: {"tag": "tune_full_cudnn", "arm": "diffusers", "knob": "full_cudnn", "round": 0}
  2026-10-22 05:58:54 retry: {"tag": "tune_full_fa3", "arm": "diffusers", "knob": "full_fa3", "round": 0}
  2026-10-22 05:59:18 retry: {"tag": "tune_full_fa3_vae", "arm": "diffusers", "knob": "full_fa3_vae", "round": 0}
  2026-10-22 05:59:30 tuned_config: {"engine": "diffusers", "config": "eager", "candidates": [[0.7901608114958287, "eager"]], "rule": "highest geometric-mean throughput over the tune points"}
  2026-10-22 06:22:51 retry: {"tag": "tune_regional_dyn_fa3", "arm": "diffusers", "knob": "regional_dyn_fa3", "round": 0}
  2026-10-22 06:25:51 retry: {"tag": "tune_regional_dyn_fa3_vae", "arm": "diffusers", "knob": "regional_dyn_fa3_vae", "round": 0}
  2026-10-22 06:26:18 tuned_config: {"engine": "diffusers", "config": "regional_dyn", "candidates": [[1.0580050236205216, "regional_dyn"], [1.0374283827261725, "regional_dyn_cudnn"], [0.7901608114958287, "eager"]], "rule": "highest geometric-mean throughput over the tune points"}
  2026-10-22 06:35:32 gate: {"verdicts": {"mstar_pr": "PASS", "mstar_pr_b": "PASS", "mstar_main": "PASS", "sglang__base": "FAIL", "sglang__base_cell": "FAIL", "sglang__bcg": "FAIL", "sglang__bcg_b8": "FAIL", "sglang__bcg_b8_cell": "FAIL", "sglang__bcg_cell": "FAIL", "sglang__compile": "FAIL", "sglang__compile_b8": "FAIL", "sgl
  2026-10-22 06:35:32 tuned_config: {"engine": "vllmomni", "config": "base", "gate_pass": false, "candidates": [[1.0989614038661881, "base", "FAIL"], [1.0948723641115043, "full", "FAIL"], [1.0751886160966018, "cudnn", "FAIL"], [0.9857838671254553, "eager", "FAIL"]], "rule": "best tune score among gate-passing knob sets"}
  2026-10-22 06:35:32 tuned_config: {"engine": "sglang", "config": "bcg", "gate_pass": false, "candidates": [[1.2708162867211232, "bcg", "FAIL"], [1.267293883834288, "bcg_b8", "FAIL"], [1.2646462959218068, "base", "FAIL"], [1.2261863428801187, "compile", "FAIL"]], "rule": "best tune score among gate-passing knob sets"}
  2026-10-22 06:53:36 gate: {"verdicts": {"mstar_pr": "PASS", "mstar_pr_b": "PASS", "mstar_main": "PASS", "sglang__base": "PASS", "sglang__bcg": "PASS", "sglang__bcg_b8": "PASS", "sglang__compile": "PASS", "sglang__compile_b8": "PASS", "vllmomni__base": "PASS", "vllmomni__cudnn": "PASS", "vllmomni__eager": "PASS", "vllmomni__f
  2026-10-22 06:53:36 tuned_config: {"engine": "vllmomni", "config": "base", "gate_pass": true, "candidates": [[1.0989614038661881, "base", "PASS"], [1.0948723641115043, "full", "PASS"], [1.0751886160966018, "cudnn", "PASS"], [0.9857838671254553, "eager", "PASS"]], "rule": "best tune score
```

</details>

## FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B), text to image

**Workload.** 1024x1024, 4 steps, no CFG (distilled; guidance_scale 1.0 sent, ignored), FlowMatch Euler, bf16, 64 PartiPrompts.

Queue phases completed: collect, gallery, gate, knobs, profile, ref_c, rounds, select, setup, summarize, tune.

### Correctness gate (vs diffusers eager reference, same prompts and seeds)

Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR 32.95 dB, p10 PSNR 26.40 dB, median SSIM 0.9847; ref_a CLIP 27.62; different-seed control median PSNR 10.46 dB; competitor same-image threshold 21.71 dB (amended gate, see dgate.py)

| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | sanity gate | parity (annotated) | parity tier | original pre-registered verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| ref_b | 34.38 | 29.02 | 24.71 | 0.9903 | 27.61 | 0 | yardstick | - | - | - |
| ref_k | 34.50 | 28.85 | 23.35 | 0.9885 | 27.61 | 0 | yardstick | - | - | - |
| ref_c | 32.95 | 26.40 | 18.54 | 0.9847 | 27.54 | 0 | yardstick | - | - | - |
| ref_seed | 10.46 | 8.56 | 6.47 | 0.3942 | 27.37 | 0 | yardstick | - | - | - |
| mstar_pr | 34.98 | 28.57 | 25.70 | 0.9902 | 27.61 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_pr_b | 34.98 | 28.57 | 25.70 | 0.9902 | 27.61 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_main | 34.99 | 28.57 | 25.70 | 0.9902 | 27.60 | 0 | PASS | yes | parity (M*) | PASS |
| sglang__base | 22.46 | 17.48 | 14.32 | 0.8666 | 27.56 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__bcg | 22.46 | 17.48 | 14.32 | 0.8666 | 27.56 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__bcg_b8 | 22.46 | 17.48 | 14.32 | 0.8666 | 27.56 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__compile | 22.20 | 17.21 | 14.38 | 0.8652 | 27.49 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__compile_b8 | 22.23 | 17.31 | 14.71 | 0.8674 | 27.53 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__base | 23.03 | 18.59 | 15.46 | 0.9176 | 27.59 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__cudnn | 23.35 | 18.65 | 15.49 | 0.9219 | 27.60 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__eager | 23.22 | 18.59 | 15.37 | 0.9221 | 27.63 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__full | 22.94 | 18.40 | 15.85 | 0.9189 | 27.62 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| ref_a | - | - | - | - | 27.62 | 0 | reference | - | - | - |


Tags: ref_b = reference in a second process; ref_k = memory-efficient SDPA; ref_c = diffusers with the tuned arm's compile/attention; ref_seed = reference with other seeds (different noise); mstar_pr_b = a second M* PR server process (same-code control); `<engine>__<knobs>` = each competitor knob set from the tune pass. Contact sheet: `sheet_klein4b.jpg` in the results folder.

### Competitor tuning (one server process per knob set, GPU 0 alone; score = geometric mean throughput over the tune points, img/s)

| engine | knob set | flags | per-point throughput | B=1 latency (s) | score | gate | chosen |
|---|---|---|---|---:|---:|---|---|
| vllm-omni | base | `(defaults)` | c=1: 2.065, c=8: 2.612 | 0.460 | 2.323 | PASS |  |
| vllm-omni | cudnn | `--diffusion-attention-backend CUDNN_ATTN` | c=1: 2.058, c=8: 2.617 | 0.462 | 2.321 | PASS |  |
| vllm-omni | full | `--diffusion-compile-granularity full` | c=1: 2.111, c=8: 2.653 | 0.451 | 2.366 | PASS | yes |
| vllm-omni | eager | `--enforce-eager` | c=1: 1.958, c=8: 2.410 | 0.487 | 2.172 | PASS |  |
| SGLang | base | `--performance-mode speed` | c=1: 3.504, c=8: 3.642 | 0.273 | 3.572 | PASS | yes |
| SGLang | compile | `--performance-mode speed --enable-torch-compile true` | c=1: 3.433, c=8: 3.576 | 0.279 | 3.504 | PASS |  |
| SGLang | bcg | `--performance-mode speed --enable-breakable-cuda-graph true --warmup-resolutions 1024x1024` | c=1: 3.500, c=8: 3.621 | 0.274 | 3.560 | PASS |  |
| SGLang | compile_b8 | `--performance-mode speed --enable-torch-compile true --batching-max-size 8` | c=1: 3.209, c=8: 3.579 | 0.300 | 3.389 | PASS |  |
| SGLang | bcg_b8 | `--performance-mode speed --enable-breakable-cuda-graph true --warmup-resolutions 1024x1024 --batching-max-size 8` | c=1: 3.157, c=8: 3.594 | 0.304 | 3.368 | PASS |  |
| diffusers | eager | `--compile none` | c=1: 2.150, c=8: 2.322 | 0.465 | 2.235 | n/a (reference impl.) |  |
| diffusers | regional_dyn | `--compile regional --dynamic` | c=1: 2.698, c=8: 2.936 | 0.371 | 2.814 | n/a (reference impl.) |  |
| diffusers | regional_dyn_cudnn | `--compile regional --dynamic --attn _native_cudnn` | c=1: 2.714, c=8: 2.911 | 0.368 | 2.811 | n/a (reference impl.) |  |
| diffusers | regional_dyn_fa3 | `--compile regional --dynamic --attn _flash_3_hub` | c=1: 2.763, c=8: 2.976 | 0.362 | 2.868 | n/a (reference impl.) |  |
| diffusers | regional_dyn_fa3_vae | `--compile regional --dynamic --attn _flash_3_hub --compile-vae` | c=1: 3.222, c=8: 3.328 | 0.310 | 3.275 | n/a (reference impl.) | yes |
| diffusers | full_dyn | `--compile full --dynamic` | c=1: 2.737, c=8: 2.926 | 0.365 | 2.830 | n/a (reference impl.) |  |
| diffusers | full_dyn_fa3 | `--compile full --dynamic --attn _flash_3_hub` | c=1: 2.759, c=8: 2.955 | 0.362 | 2.855 | n/a (reference impl.) |  |

### Exact server commands (round 1)

```
/root/work/mstar-main/.venv/bin/python -m mstar.cli.main serve flux2_klein --config /root/work/mstar-main/configs/flux2_klein.yaml --gpus 0 --port 8400
/root/work/mstar/.venv/bin/python -m mstar.cli.main serve flux2_klein --config /root/work/mstar/configs/flux2_klein.yaml --gpus 0 --port 8400
/root/work/engines/sgld/bin/sglang serve --model-path black-forest-labs/FLUX.2-klein-4B --num-gpus 1 --host 127.0.0.1 --port 8400 --scheduler-port 8500 --performance-mode speed
/root/work/engines/vllmomni/bin/vllm serve black-forest-labs/FLUX.2-klein-4B --omni --host 127.0.0.1 --port 8400 --diffusion-compile-granularity full
```

Versions: diffusers: {"packages": ["diffusers==0.41.0", "kernels==0.17.2", "torch==2.14.1+cu132", "transformers==5.20.2", "triton==3.8.0"]}; mstar-main: {"imports": "/root/work/mstar-main/mstar/__init__.py 2.12.1+cu132 0.41.0", "commit": "HEAD abdfd580296a843f62ce967ddb93b0a3dfb225e3"}; mstar-pr: {"imports": "/root/work/mstar/mstar/__init__.py 2.12.1+cu132 0.41.0", "commit": "perf/overnight-all d248477ec795c6ddf0716e23932be15be2b573e2"}; sglang: {"packages": ["cache-dit==1.5.1", "diffusers==0.37.0", "flash-attn-4==4.0.0b36", "flashinfer-python==0.7.1", "kernels==0.14.1", "kernels-data==0.16.2", "sgl-deep-ep==0.1.2.post1", "sgl-deep-gemm==0.2.2", "sglang==0.5.22", "sglang-kernel==0.4.9", "torch==2.14.1+cu132", "transformers==5.19.0", "triton; vllmomni: {"packages": ["cache-dit==1.5.0", "diffusers==0.40.0", "fa3-fwd==0.0.3", "flashinfer-python==0.6.18.post1", "kernels==0.16.1", "kernels-data==0.16.2", "torch==2.13.0+cu132", "transformers==5.14.1", "triton==3.7.1", "vllm==0.30.0", "vllm-omni==0.30.0"]}

### Timed results (3 rounds, fresh server per round, rotated order)

Arm configs: M* PR = config; M* main = config; vllm-omni = full; SGLang = base; diffusers (tuned, offline batch) = regional_dyn_fa3_vae

## Throughput (img/s), steps=4: mean [min-max] over rounds; n rounds; failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* PR | 2.719 [2.706-2.736] n=3 | 3.294 [3.285-3.311] n=3 | 3.388 [3.384-3.394] n=3 | 3.385 [3.365-3.394] n=3 | 3.416 [3.398-3.426] n=3 |
| M* main | 2.644 [2.639-2.648] n=3 | 3.116 [3.056-3.189] n=3 | 3.306 [3.288-3.320] n=3 | 3.330 [3.318-3.351] n=3 | 3.361 [3.357-3.369] n=3 |
| vllm-omni | 2.094 [2.089-2.100] n=3 | 2.629 [2.615-2.641] n=3 | 2.668 [2.667-2.668] n=3 | 2.674 [2.672-2.678] n=3 | 2.672 [2.667-2.678] n=3 |
| SGLang | 3.522 [3.520-3.528] n=3 | 3.636 [3.635-3.638] n=3 | 3.635 [3.633-3.637] n=3 | 3.629 [3.626-3.635] n=3 | 3.624 [3.623-3.625] n=3 |
| diffusers (tuned, offline batch) | 3.226 [3.209-3.236] n=3 | 3.301 [3.299-3.303] n=3 | 3.332 [3.329-3.335] n=3 | 3.324 [3.320-3.328] n=3 | 3.326 [3.325-3.326] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* main | 1.03x | 1.06x | 1.02x | 1.02x | 1.02x |
| vllm-omni | 1.30x | 1.25x | 1.27x | 1.27x | 1.28x |
| SGLang | 0.77x | 0.91x | 0.93x | 0.93x | 0.94x |
| diffusers (tuned, offline batch) | 0.84x | 1.00x | 1.02x | 1.02x | 1.03x |

## B=1 latency (s), steps=4, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 0.338 [0.337-0.340] | 0.338 | 0.352 | 32 | 3 | 0 |
| M* main | 0.349 [0.349-0.350] | 0.349 | 0.361 | 32 | 3 | 0 |
| vllm-omni | 0.454 [0.452-0.455] | 0.454 | 0.466 | 32 | 3 | 0 |
| SGLang | 0.273 [0.272-0.273] | 0.273 | 0.277 | 32 | 3 | 0 |
| diffusers (tuned, offline batch) | 0.310 [0.309-0.312] | 0.310 | 0.317 | 32 | 3 | 0 |

1-min load average at cell start: median 4.0, max 16.6 over 75 cells (192 vCPUs).

### Profile (phases_c1, MSTAR_PHASE_TIMING=100, benchmark/worker_phases): GPU-bound (worker waits on the GPU most of each step); event_sync / iter_total up to 108%

<details><summary>phase tables</summary>

```
=== worker_0  iters 100-200  records=2  batch=1.00 (range 1.0-1.0) ===
    work: 200 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    0.794    0.000    4.170      199  790200.0 
engine.collect_outputs                       0.030    0.030    0.040      172   25800.0 
engine.declare_step                          0.000    0.000    0.000      172       0.0 
engine.forward                               1.088    0.290    4.180      172  935400.0 
engine.postprocess                           0.000    0.000    0.000      200       0.0 
engine.preprocess                            0.085    0.080    0.140      172   73050.0 
gpu_exec                                    40.429   57.555   62.850      19840025100.0 
iter_total                                  36.255   60.845   65.020      20036255000.0 
rust.send_outputs.call                       0.025    0.020    0.080      171   21350.0 
rust.send_outputs.marshal                    0.025    0.025    0.070      171   21350.0 
speculate                                    0.080    0.080    0.160      114   45600.0 
submit_spec                                  0.320    0.350    0.530      142  227100.0 
worker.await_admit_settled                   3.735    3.675    7.250       58 1083150.0 
worker.gpu_thread.await_plan                 0.010    0.000    0.030      142    7100.0 
worker.gpu_thread.exec                       1.123    0.490    4.300      200 1123250.0 
worker.gpu_thread.prepare_inputs             1.035    0.060    6.830      200 1035150.0 
worker.postprocess.check_stop                0.205    0.155    0.310      171  175100.0 
worker.postprocess.check_stop_tail           0.010    0.010    0.030      171    8550.0 
worker.postprocess.cleanup_inputs            0.010    0.010    0.010      199    9950.0 
worker.postprocess.completion_event_sync    45.613   57.965   62.020      17138999350.0 
worker.postprocess.event_sync               39.188   56.480   62.020      19938991650.0 
worker.postprocess.pending_loop_stops        0.010    0.010    0.010      171    8550.0 
worker.postprocess.prematerialize            0.195    0.145    0.270      171  166550.0 
worker.postprocess.register_outputs          0.120    0.000    0.270      171  102600.0 
worker.postprocess.route                     0.030    0.030    0.050      171   25650.0 
worker.postprocess.route_outputs             0.095    0.090    0.150      171   81200.0 
worker.postprocess.send                      0.055    0.045    0.130      171   47000.0 
worker.postprocess.send_outputs              0.065    0.060    0.140      171   55550.0 
worker.postprocess.send_prep                 0.010    0.010    0.010      171    8550.0 
worker.postprocess.stop_loops                0.005    0.000    0.020      171    4250.0 
worker.postprocess.store_tensors             0.055    0.050    0.090      171   47000.0 
worker.postprocess.update_lru                0.000    0.000    0.000      171       0.0 
worker.postprocess_batch                    39.637   56.785   62.370      19939439250.0 
worker.schedule                              0.012    0.005    0.080      899   54950.0*
worker.schedule_yield_away                   0.030    0.010    0.090       85   12750.0 
worker.submit_spec.launch_wait               0.280    0.325    0.510      142  198700.0 
    * sampled on passes that ran no batch, so its time is spread over work it did not all do: us/1k is an upper bound for this row
```

</details>

### Summary (generated from the tables above)

- vs M* main: throughput 1.02x (c=16) to 1.06x (c=2); B=1 latency 0.338 s vs 0.349 s (0.97x). M* PR is not slower at any point.
- vs vllm-omni: throughput 1.25x (c=2) to 1.30x (c=1); B=1 latency 0.338 s vs 0.454 s (0.75x). M* PR is not slower at any point.
- vs SGLang: throughput 0.77x (c=1) to 0.94x (c=16); B=1 latency 0.338 s vs 0.273 s (1.24x). M* PR is slower at c=1 (0.77x), c=2 (0.91x), c=4 (0.93x), c=8 (0.93x), c=16 (0.94x).
- vs diffusers (offline batch, no serving): throughput 0.84x (c=1) to 1.03x (c=16); B=1 latency 0.338 s vs 0.310 s (1.09x). M* PR is slower at c=1 (0.84x), c=2 (1.00x).

<details><summary>notes.txt (startup times, health checks, decisions)</summary>

```
notes for klein4b (2026-10-22 09:54 UTC)

server startup seconds per arm (all starts, incl. cold compiles): vllmomni: min 18 median 35 max 72 n=8; sglang: min 33 median 33 max 126 n=9; mstar-pr: min 54 median 57 max 788 n=7; mstar-main: min 96 median 110 max 334 n=4

health (per server log):
  main_mstar-main_config_r1: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-main_config_r2: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-main_config_r3: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r1: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r2: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r3: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_sglang_base_r1: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_sglang_base_r2: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_sglang_base_r3: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_vllmomni_full_r1: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  main_vllmomni_full_r2: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  main_vllmomni_full_r3: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  tune_base_sglang_base_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_base_vllmomni_base_r0: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_bcg_b8_sglang_bcg_b8_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_bcg_sglang_bcg_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_compile_b8_sglang_compile_b8_r0: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-22 08:36:57] Configuring layerwise offload for Flux2Transformer2DModel: transforme"], "errors": 0}
  tune_compile_sglang_compile_r0: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-22 08:31:54] Configuring layerwise offload for Flux2Transformer2DModel: transforme"], "errors": 0}
  tune_cudnn_vllmomni_cudnn_r0: {"attention": ["CUDNN_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_eager_vllmomni_eager_r0: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  tune_full_vllmomni_full_r0: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}

decisions:
  2026-10-22 08:59:55 tuned_config: {"engine": "diffusers", "config": "regional_dyn_fa3_vae", "candidates": [[3.2746882737484446, "regional_dyn_fa3_vae"], [2.8678508730668173, "regional_dyn_fa3"], [2.855378905614308, "full_dyn_fa3"], [2.8295285130558128, "full_dyn"], [2.8142014247566354, "regional_dyn"], [2.8105409335104814, "regional
  2026-10-22 09:08:30 gate: {"verdicts": {"mstar_pr": "PASS", "mstar_pr_b": "PASS", "mstar_main": "PASS", "sglang__base": "PASS", "sglang__bcg": "PASS", "sglang__bcg_b8": "PASS", "sglang__compile": "PASS", "sglang__compile_b8": "PASS", "vllmomni__base": "PASS", "vllmomni__cudnn": "PASS", "vllmomni__eager": "PASS", "vllmomni__f
  2026-10-22 09:08:31 tuned_config: {"engine": "vllmomni", "config": "full", "gate_pass": true, "candidates": [[2.366205198350116, "full", "PASS"], [2.3226184271418227, "base", "PASS"], [2.3206887244172965, "cudnn", "PASS"], [2.172302633085855, "eager", "PASS"]], "rule": "best tune score among gate-passing knob sets"}
  2026-10-22 09:08:31 tuned_config: {"engine": "sglang", "config": "base", "gate_pass": true, "candidates": [[3.572452741266595, "base", "PASS"], [3.5601921589450716, "bcg", "PASS"], [3.503856708352505, "compile", "PASS"], [3.3890299185256696, "compile_b8", "PASS"], [3.3681961143541135, "bcg_b8", "PASS"]], "rule": "best tune score amo
  2026-10-22 09:53:08 profile_points: {"points": [1], "ratios": {"1": 0.7720494871228353, "2": 0.9059189974497223, "4": 0.9318920039904629, "8": 0.9325824030462496, "16": 0.9427527559748942}, "rule": "point where mstar-pr / best competitor is lowest, plus c=1 if mstar-pr trails there"}
```

</details>

## Wan2.2-TI2V-5B (Wan-AI/Wan2.2-TI2V-5B-Diffusers), text to video

**Workload.** 1280x704 x 81 frames (24 fps), guidance 5.0 (CFG), empty negative prompt, UniPC flow_shift 5.0 (checkpoint scheduler), bf16; 20 steps for the sweep, 50 steps for the B=1 latency row; 16 VBench prompts (gate: first 4).

Queue phases completed: collect, gallery, gate, knobs, profile, ref_c, rounds, select, setup, summarize, tune.

### Correctness gate (vs diffusers eager reference, same prompts and seeds)

Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR 28.52 dB, p10 PSNR 22.10 dB, median SSIM 0.9306; ref_a CLIP 21.28; different-seed control median PSNR 9.62 dB; competitor same-image threshold 19.07 dB (amended gate, see dgate.py)

| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | sanity gate | parity (annotated) | parity tier | original pre-registered verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| ref_b | 41.16 | 36.01 | 34.08 | 0.9791 | 21.34 | 0 | yardstick | - | - | - |
| ref_k | 30.62 | 23.85 | 21.16 | 0.9547 | 21.38 | 0 | yardstick | - | - | - |
| ref_c | 28.52 | 22.10 | 20.44 | 0.9306 | 21.40 | 0 | yardstick | - | - | - |
| ref_seed | 9.62 | 7.94 | 7.63 | 0.4598 | 21.32 | 0 | yardstick | - | - | - |
| mstar_pr | 28.09 | 20.42 | 18.05 | 0.9309 | 21.60 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_pr_b | 28.09 | 20.42 | 18.05 | 0.9309 | 21.60 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_main | 28.67 | 20.74 | 18.03 | 0.9357 | 21.65 | 0 | PASS | yes | parity (M*) | PASS |
| sglang__base | 27.04 | 23.18 | 22.76 | 0.9183 | 21.47 | 0 | PASS | yes | same-image (competitor, amended) | PASS |
| sglang__bcg | 27.04 | 23.18 | 22.76 | 0.9183 | 21.47 | 0 | PASS | yes | same-image (competitor, amended) | PASS |
| sglang__compile | 27.09 | 23.72 | 23.46 | 0.9194 | 22.16 | 0 | PASS | yes | same-image (competitor, amended) | PASS |
| sglang__compile_b4 | 27.09 | 23.72 | 23.46 | 0.9194 | 22.16 | 0 | PASS | yes | same-image (competitor, amended) | PASS |
| vllmomni__b4 | 11.65 | 8.60 | 7.35 | 0.4927 | 23.43 | 0 | PASS | no (annotated) | same-image (competitor, amended) | PASS (distribution-level only) |
| vllmomni__base | 11.65 | 8.60 | 7.35 | 0.4927 | 23.43 | 0 | PASS | no (annotated) | same-image (competitor, amended) | PASS (distribution-level only) |
| vllmomni__cudnn | 11.67 | 8.62 | 7.35 | 0.4940 | 22.82 | 0 | PASS | no (annotated) | same-image (competitor, amended) | PASS (distribution-level only) |
| vllmomni__full | 11.73 | 8.61 | 7.34 | 0.4944 | 22.81 | 0 | PASS | no (annotated) | same-image (competitor, amended) | PASS (distribution-level only) |
| ref_a | - | - | - | - | 21.28 | 0 | reference | - | - | - |


Tags: ref_b = reference in a second process; ref_k = memory-efficient SDPA; ref_c = diffusers with the tuned arm's compile/attention; ref_seed = reference with other seeds (different noise); mstar_pr_b = a second M* PR server process (same-code control); `<engine>__<knobs>` = each competitor knob set from the tune pass. Contact sheet: `sheet_wan22.jpg` in the results folder.

### Competitor tuning (one server process per knob set, GPU 0 alone; score = geometric mean throughput over the tune points, videos/s)

| engine | knob set | flags | per-point throughput | B=1 latency (s) | score | gate | chosen |
|---|---|---|---|---:|---:|---|---|
| vllm-omni | base | `(defaults)` | c=1: 0.043, c=4: 0.044 | 23.007 | 0.044 | PASS |  |
| vllm-omni | cudnn | `--diffusion-attention-backend CUDNN_ATTN` | c=1: 0.041, c=4: 0.042 | 24.406 | 0.041 | PASS |  |
| vllm-omni | full | `--diffusion-compile-granularity full` | c=1: 0.044, c=4: 0.045 | 22.710 | 0.044 | PASS | yes |
| vllm-omni | b4 | `--max-num-seqs 4` | c=1: 0.043, c=4: 0.044 | 23.029 | 0.044 | PASS |  |
| SGLang | base | `--performance-mode speed` | c=1: 0.041, c=4: 0.041 | 24.502 | 0.041 | PASS | yes |
| SGLang | compile | `--performance-mode speed --enable-torch-compile true` | c=1: 0.039, c=4: 0.039 | 25.260 | 0.039 | PASS |  |
| SGLang | bcg | `--performance-mode speed --enable-breakable-cuda-graph true --warmup-resolutions 1280x704 --warmup-num-frames 81` | c=1: 0.040, c=4: 0.041 | 24.654 | 0.040 | PASS |  |
| SGLang | compile_b4 | `--performance-mode speed --enable-torch-compile true --batching-max-size 4` | c=1: 0.040, c=4: 0.039 | 25.200 | 0.039 | PASS |  |
| diffusers | eager | `--compile none` | c=1: 0.030, c=4: 0.030 | 33.652 | 0.030 | n/a (reference impl.) |  |
| diffusers | regional_dyn | `--compile regional --dynamic` | c=1: 0.038, c=4: 0.037 | 26.269 | 0.038 | n/a (reference impl.) |  |
| diffusers | regional_dyn_fa3 | `--compile regional --dynamic --attn _flash_3_hub` | c=1: 0.040, c=4: 0.040 | 24.868 | 0.040 | n/a (reference impl.) | yes |
| diffusers | full_dyn | `--compile full --dynamic` | c=1: 0.037, c=4: 0.037 | 26.728 | 0.037 | n/a (reference impl.) |  |

### Exact server commands (round 1)

```
/root/work/mstar-main/.venv/bin/python -m mstar.cli.main serve wan22 --config /root/work/mstar-main/configs/wan22.yaml --gpus 0 --port 8400
/root/work/mstar/.venv/bin/python -m mstar.cli.main serve wan22 --config /root/work/mstar/configs/wan22.yaml --gpus 0 --port 8400
/root/work/engines/sgld/bin/sglang serve --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers --num-gpus 1 --host 127.0.0.1 --port 8400 --scheduler-port 8500 --performance-mode speed
/root/work/engines/vllmomni/bin/vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni --host 127.0.0.1 --port 8400 --diffusion-compile-granularity full
```

Versions: diffusers: {"packages": ["diffusers==0.41.0", "kernels==0.17.2", "torch==2.14.1+cu132", "transformers==5.20.2", "triton==3.8.0"]}; mstar-main: {"imports": "/root/work/mstar-main/mstar/__init__.py 2.12.1+cu132 0.41.0", "commit": "HEAD abdfd580296a843f62ce967ddb93b0a3dfb225e3"}; mstar-pr: {"imports": "/root/work/mstar/mstar/__init__.py 2.12.1+cu132 0.41.0", "commit": "perf/overnight-all d248477ec795c6ddf0716e23932be15be2b573e2"}; sglang: {"packages": ["cache-dit==1.5.1", "diffusers==0.37.0", "flash-attn-4==4.0.0b36", "flashinfer-python==0.7.1", "kernels==0.14.1", "kernels-data==0.16.2", "sgl-deep-ep==0.1.2.post1", "sgl-deep-gemm==0.2.2", "sglang==0.5.22", "sglang-kernel==0.4.9", "torch==2.14.1+cu132", "transformers==5.19.0", "triton; vllmomni: {"packages": ["cache-dit==1.5.0", "diffusers==0.40.0", "fa3-fwd==0.0.3", "flashinfer-python==0.6.18.post1", "kernels==0.16.1", "kernels-data==0.16.2", "torch==2.13.0+cu132", "transformers==5.14.1", "triton==3.7.1", "vllm==0.30.0", "vllm-omni==0.30.0"]}

### Timed results (3 rounds, fresh server per round, rotated order)

Arm configs: M* PR = config; M* main = config; vllm-omni = full; SGLang = base; diffusers (tuned, offline batch) = regional_dyn_fa3

## Throughput (videos/min), steps=20: mean [min-max] over rounds; n rounds; failed

| arm | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| M* PR | 2.667 [2.664-2.670] n=3 | 2.712 [2.710-2.713] n=3 | 2.724 [2.722-2.725] n=3 | 2.730 [2.729-2.730] n=3 |
| M* main | 2.433 [2.430-2.435] n=3 | 2.487 [2.486-2.489] n=3 | 2.501 [2.500-2.502] n=3 | 2.505 [2.504-2.506] n=3 |
| vllm-omni | 2.632 [2.632-2.632] n=3 | 2.685 [2.683-2.687] n=3 | 2.693 [2.692-2.693] n=3 | 2.697 [2.696-2.698] n=3 |
| SGLang | 2.424 [2.424-2.425] n=3 | 2.439 [2.438-2.441] n=3 | 2.440 [2.439-2.441] n=3 | 2.442 [2.441-2.444] n=3 |
| diffusers (tuned, offline batch) | 2.409 [2.407-2.411] n=3 | 2.402 [2.400-2.403] n=3 | 2.405 [2.404-2.406] n=3 | 2.399 [2.397-2.401] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| M* main | 1.10x | 1.09x | 1.09x | 1.09x |
| vllm-omni | 1.01x | 1.01x | 1.01x | 1.01x |
| SGLang | 1.10x | 1.11x | 1.12x | 1.12x |
| diffusers (tuned, offline batch) | 1.11x | 1.13x | 1.13x | 1.14x |

## B=1 latency (s), steps=20, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 22.434 [22.411-22.461] | 22.362 | 22.610 | 3 | 3 | 0 |
| M* main | 24.608 [24.587-24.631] | 24.519 | 24.813 | 3 | 3 | 0 |
| vllm-omni | 22.726 [22.720-22.730] | 22.727 | 22.770 | 3 | 3 | 0 |
| SGLang | 24.577 [24.563-24.601] | 24.599 | 24.701 | 3 | 3 | 0 |
| diffusers (tuned, offline batch) | 24.910 [24.888-24.931] | 24.902 | 24.960 | 3 | 3 | 0 |

## Throughput (videos/min), steps=50: mean [min-max] over rounds; n rounds; failed

| arm | c=1 |
|---|---:|
| M* PR | 1.156 [1.155-1.157] n=3 |
| M* main | 1.053 [1.052-1.054] n=3 |
| vllm-omni | 1.160 [1.159-1.161] n=3 |
| SGLang | 1.140 [1.138-1.141] n=3 |
| diffusers (tuned, offline batch) | 1.106 [1.105-1.106] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 |
|---|---:|
| M* main | 1.10x |
| vllm-omni | 1.00x |
| SGLang | 1.01x |
| diffusers (tuned, offline batch) | 1.05x |

## B=1 latency (s), steps=50, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 51.839 [51.791-51.879] | 51.785 | 51.994 | 3 | 3 | 0 |
| M* main | 56.937 [56.883-56.986] | 56.894 | 57.095 | 3 | 3 | 0 |
| vllm-omni | 51.645 [51.604-51.678] | 51.658 | 51.680 | 3 | 3 | 0 |
| SGLang | 52.473 [52.419-52.545] | 52.450 | 52.551 | 3 | 3 | 0 |
| diffusers (tuned, offline batch) | 54.267 [54.245-54.285] | 54.318 | 54.352 | 3 | 3 | 0 |

1-min load average at cell start: median 1.7, max 7.2 over 75 cells (192 vCPUs).

### Profile (phases_c2, MSTAR_PHASE_TIMING=100, benchmark/worker_phases): CPU-bound; event_sync / iter_total up to 2%

<details><summary>phase tables</summary>

```
=== worker_0  iters 100-100  records=1  batch=1.00 (range 1.0-1.0) ===
    work: 100 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                 1053.950  969.010 1773.230       991043410500.0 
engine.admit                                 0.010    0.010    0.010       86    8600.0 
engine.collect_outputs                       0.060    0.060    0.090       95   57000.0 
engine.commit                                0.010    0.010    0.020       85    8500.0 
engine.declare_step                          0.010    0.010    0.030       96    9600.0 
engine.forward                            1101.090  971.580 1769.030       951046035500.0 
engine.plan.fresh                            9.540    0.870    1.140       86 8204400.0 
engine.postprocess                           0.000    0.000    0.010       99       0.0 
engine.preprocess                            0.000    0.000    0.000       96       0.0 
gpu_exec                                  1087.880  972.580 2335.060       991077001200.0 
iter_total                                1054.150  972.260 2346.210      1001054150000.0 
res.admit.dit_self_attn                      0.000    0.000    0.000       86       0.0 
res.commit.dit_self_attn                     0.000    0.000    0.010       85       0.0 
res.plan.dit_self_attn                       9.540    0.870    1.140       86 8204400.0 
rust.send_outputs.call                       0.020    0.010    0.060       95   19000.0 
rust.send_outputs.marshal                    0.010    0.000    0.080       95    9500.0 
speculate                                    0.090    0.090    0.130       83   74700.0 
submit_spec                                  1.170    1.060    1.460       96 1123200.0 
worker.await_admit_settled                 385.950  758.240  758.240        2 7719000.0 
worker.gpu_thread.await_plan                 0.000    0.000    0.000       96       0.0 
worker.gpu_thread.exec                    1065.050  972.560 1769.150       991054399500.0 
worker.gpu_thread.prepare_inputs            17.160    0.030   12.480      10017160000.0 
worker.postprocess.check_stop                1.950    1.320    4.620       95 1852500.0 
worker.postprocess.check_stop_tail           0.010    0.010    0.020       95    9500.0 
worker.postprocess.cleanup_inputs            0.020    0.020    0.030       99   19800.0 
worker.postprocess.completion_event_sync    23.590    0.120    2.450       9522410500.0 
worker.postprocess.event_sync               22.630    0.120    2.450       9922403700.0 
worker.postprocess.pending_loop_stops        0.010    0.010    0.010       95    9500.0 
worker.postprocess.prematerialize            1.950    1.310    4.620       95 1852500.0 
worker.postprocess.register_outputs          0.870    0.000    0.020       95  826500.0 
worker.postprocess.route                     0.050    0.040    0.070       95   47500.0 
worker.postprocess.route_outputs             0.140    0.130    0.220       95  133000.0 
worker.postprocess.send                      0.030    0.010    0.150       95   28500.0 
worker.postprocess.send_outputs              0.050    0.030    0.180       95   47500.0 
worker.postprocess.send_prep                 0.010    0.010    0.020       95    9500.0 
worker.postprocess.stop_loops                0.000    0.000    0.010       95       0.0 
worker.postprocess.store_tensors             0.080    0.080    0.130       95   76000.0 
worker.postprocess.update_lru                0.000    0.000    0.000       95       0.0 
worker.postprocess_batch                    25.580    1.630   41.370       9925324200.0 
worker.schedule                              0.010    0.000    0.010      596   59600.0*
worker.schedule_yield_away                   0.080    0.100    0.140       16   12800.0 
worker.submit_spec.launch_wait               1.140    1.030    1.410       96 1094400.0 
    * sampled on passes that ran no batch, so its time is spread over work it did not all do: us/1k is an upper bound for this row
```

</details>

### Summary (generated from the tables above)

- vs M* main: throughput 1.09x (c=4) to 1.10x (c=1); B=1 latency 22.434 s vs 24.608 s (0.91x). M* PR is not slower at any point.
- vs vllm-omni: throughput 1.01x (c=2) to 1.01x (c=1); B=1 latency 22.434 s vs 22.726 s (0.99x). M* PR is not slower at any point.
- vs SGLang: throughput 1.10x (c=1) to 1.12x (c=8); B=1 latency 22.434 s vs 24.577 s (0.91x). M* PR is not slower at any point.
- vs diffusers (offline batch, no serving): throughput 1.11x (c=1) to 1.14x (c=8); B=1 latency 22.434 s vs 24.910 s (0.90x). M* PR is not slower at any point.

<details><summary>notes.txt (startup times, health checks, decisions)</summary>

```
notes for wan22 (2026-10-22 17:47 UTC)

server startup seconds per arm (all starts, incl. cold compiles): mstar-pr: min 21 median 21 max 24 n=9; vllmomni: min 24 median 48 max 117 n=8; sglang: min 39 median 41 max 220 n=8; mstar-main: min 21 median 21 max 24 n=5

health (per server log):
  main_mstar-main_config_r1: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-main_config_r2: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-main_config_r3: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r1: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r2: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r3: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_sglang_base_r1: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_sglang_base_r2: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_sglang_base_r3: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  main_vllmomni_full_r1: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  main_vllmomni_full_r2: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  main_vllmomni_full_r3: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  tune_b4_vllmomni_b4_r0: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_base_sglang_base_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_base_vllmomni_base_r0: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_bcg_sglang_bcg_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_compile_b4_sglang_compile_b4_r0: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-22 10:59:58] Configuring layerwise offload for WanTransformer3DModel: blocks (30 l"], "errors": 0}
  tune_compile_sglang_compile_r0: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-22 10:39:03] Configuring layerwise offload for WanTransformer3DModel: blocks (30 l"], "errors": 0}
  tune_cudnn_vllmomni_cudnn_r0: {"attention": ["CUDNN_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_full_vllmomni_full_r0: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}

decisions:
  2026-10-22 11:40:58 tuned_config: {"engine": "diffusers", "config": "regional_dyn_fa3", "candidates": [[0.040164615028870004, "regional_dyn_fa3"], [0.03772313187009695, "regional_dyn"], [0.037337515352328224, "full_dyn"], [0.029701240862561595, "eager"]], "rule": "highest geometric-mean throughput over the tune points"}
  2026-10-22 11:47:11 gate: {"verdicts": {"mstar_pr": "PASS", "mstar_pr_b": "PASS", "mstar_main": "PASS", "sglang__base": "PASS", "sglang__bcg": "PASS", "sglang__compile": "PASS", "sglang__compile_b4": "PASS", "vllmomni__b4": "FAIL (broken output)", "vllmomni__base": "FAIL (broken output)", "vllmomni__cudnn": "FAIL (broken out
  2026-10-22 11:47:11 tuned_config: {"engine": "vllmomni", "config": "full", "gate_pass": false, "candidates": [[0.04429865837556845, "full", "FAIL (broken output)"], [0.04373368600530884, "base", "FAIL (broken output)"], [0.04351835568561748, "b4", "FAIL (broken output)"], [0.041236963899148715, "cudnn", "FAIL (broken output)"]], "ru
  2026-10-22 11:47:11 tuned_config: {"engine": "sglang", "config": "base", "gate_pass": true, "candidates": [[0.04065829219261334, "base", "PASS"], [0.04044882769407943, "bcg", "PASS"], [0.03946112817261297, "compile_b4", "PASS"], [0.03939869581619588, "compile", "PASS"]], "rule": "best tune score among gate-passing knob sets"}
  2026-10-22 11:54:04 gate: {"verdicts": {"mstar_pr": "PASS", "mstar_pr_b": "PASS", "mstar_main": "PASS", "sglang__base": "PASS", "sglang__bcg": "PASS", "sglang__compile": "PASS", "sglang__compile_b4": "PASS", "vllmomni__b4": "PASS", "vllmomni__base": "PASS", "vllmomni__cudnn": "PASS", "vllmomni__full": "PASS"}}
  2026-10-22 11:54:04 tuned_config: {"engine": "vllmomni", "config": "full", "gate_pass": true, "candidates": [[0.04429865837556845, "full", "PASS"], [0.04373368600530884, "base", "PASS"], [0.04351835568561748, "b4", "PASS"], [0.041236963899148715, "cudnn", "PASS"]], "rule": "best tune score among gate-passing knob sets"}
  2026-10-22 11:54:04 tuned_config: {"engine": "sglang", "config": "base", "gate_pass": true, "candidates": [[0.04065829219261334, "base", "PASS"], [0.04044882769407943, "bcg", "PASS"], [0.03946112817261297, "compile_b4", "PASS"], [0.03939869581619588, "compile", "PASS"]], "rule": "best tune score among gate-passing knob sets"}
  2026-10-22 17:43:27 profile_points: {"points": [2], "ratios": {"1": 1.0133342984323728, "2": 1.0102798670215551, "4": 1.0116936207473775, "8": 1.0119635288267628}, "rule": "point where mstar-pr / best competitor is lowest, plus c=1 if mstar-pr trails there"}
```

</details>

## FLUX.2 [klein] 9B (black-forest-labs/FLUX.2-klein-9B), text to image

**Workload.** 1024x1024, 4 steps, no CFG (distilled), FlowMatch Euler, bf16, 64 PartiPrompts.

Queue phases completed: collect, gallery, gate, knobs, profile, ref_c, rounds, select, setup, tune.

### Correctness gate (vs diffusers eager reference, same prompts and seeds)

Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR 30.45 dB, p10 PSNR 24.86 dB, median SSIM 0.9763; ref_a CLIP 27.94; different-seed control median PSNR 10.65 dB; competitor same-image threshold 20.55 dB (amended gate, see dgate.py)

| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | sanity gate | parity (annotated) | parity tier | original pre-registered verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| ref_b | 32.39 | 26.51 | 22.55 | 0.9836 | 27.92 | 0 | yardstick | - | - | - |
| ref_k | 32.16 | 26.04 | 20.95 | 0.9844 | 27.90 | 0 | yardstick | - | - | - |
| ref_c | 30.45 | 24.86 | 19.61 | 0.9763 | 27.93 | 0 | yardstick | - | - | - |
| ref_seed | 10.65 | 8.26 | 6.47 | 0.3536 | 27.82 | 0 | yardstick | - | - | - |
| mstar_pr | 31.77 | 25.91 | 21.53 | 0.9833 | 27.96 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_pr_b | 31.77 | 25.91 | 21.53 | 0.9833 | 27.96 | 0 | PASS | yes | parity (M*) | PASS |
| mstar_main | 31.77 | 25.91 | 21.53 | 0.9833 | 27.95 | 0 | PASS | yes | parity (M*) | PASS |
| sglang__base | 24.37 | 20.34 | 15.51 | 0.8933 | 27.93 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__bcg | 24.37 | 20.34 | 15.51 | 0.8933 | 27.93 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__bcg_b8 | 24.37 | 20.34 | 15.51 | 0.8933 | 27.93 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__compile | 24.28 | 20.09 | 15.17 | 0.8812 | 27.91 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| sglang__compile_b8 | 24.02 | 20.13 | 15.41 | 0.8804 | 27.87 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__base | 25.40 | 21.21 | 16.29 | 0.9394 | 27.86 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__cudnn | 25.75 | 21.71 | 16.41 | 0.9421 | 27.88 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__eager | 25.28 | 21.52 | 16.34 | 0.9352 | 27.86 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| vllmomni__full | 25.25 | 20.83 | 16.61 | 0.9305 | 27.86 | 0 | PASS | yes | same-image (competitor, amended) | FAIL |
| ref_a | - | - | - | - | 27.94 | 0 | reference | - | - | - |


Tags: ref_b = reference in a second process; ref_k = memory-efficient SDPA; ref_c = diffusers with the tuned arm's compile/attention; ref_seed = reference with other seeds (different noise); mstar_pr_b = a second M* PR server process (same-code control); `<engine>__<knobs>` = each competitor knob set from the tune pass. Contact sheet: `sheet_klein9b.jpg` in the results folder.

### Competitor tuning (one server process per knob set, GPU 0 alone; score = geometric mean throughput over the tune points, img/s)

| engine | knob set | flags | per-point throughput | B=1 latency (s) | score | gate | chosen |
|---|---|---|---|---:|---:|---|---|
| vllm-omni | base | `(defaults)` | c=1: 1.348, c=8: 1.549 | 0.717 | 1.445 | PASS |  |
| vllm-omni | cudnn | `--diffusion-attention-backend CUDNN_ATTN` | c=1: 1.335, c=8: 1.545 | 0.725 | 1.436 | PASS |  |
| vllm-omni | full | `--diffusion-compile-granularity full` | c=1: 1.360, c=8: 1.568 | 0.711 | 1.460 | PASS | yes |
| vllm-omni | eager | `--enforce-eager` | c=1: 1.272, c=8: 1.453 | 0.762 | 1.359 | PASS |  |
| SGLang | base | `--performance-mode speed` | c=1: 1.706, c=8: 1.766 | 0.575 | 1.736 | PASS |  |
| SGLang | compile | `--performance-mode speed --enable-torch-compile true` | c=1: 1.716, c=8: 1.789 | 0.571 | 1.752 | PASS |  |
| SGLang | bcg | `--performance-mode speed --enable-breakable-cuda-graph true --warmup-resolutions 1024x1024` | c=1: 1.706, c=8: 1.761 | 0.575 | 1.734 | PASS |  |
| SGLang | compile_b8 | `--performance-mode speed --enable-torch-compile true --batching-max-size 8` | c=1: 1.717, c=8: 1.842 | 0.570 | 1.778 | PASS | yes |
| SGLang | bcg_b8 | `--performance-mode speed --enable-breakable-cuda-graph true --warmup-resolutions 1024x1024 --batching-max-size 8` | c=1: 1.696, c=8: 1.814 | 0.578 | 1.754 | PASS |  |
| diffusers | eager | `--compile none` | c=1: 1.276, c=8: 1.310 | 0.784 | 1.293 | n/a (reference impl.) |  |
| diffusers | regional_dyn | `--compile regional --dynamic` | c=1: 1.596, c=8: 1.654 | 0.626 | 1.625 | n/a (reference impl.) |  |
| diffusers | regional_dyn_cudnn | `--compile regional --dynamic --attn _native_cudnn` | c=1: 1.597, c=8: 1.651 | 0.626 | 1.624 | n/a (reference impl.) |  |
| diffusers | regional_dyn_fa3 | `--compile regional --dynamic --attn _flash_3_hub` | c=1: 1.624, c=8: 1.685 | 0.616 | 1.654 | n/a (reference impl.) |  |
| diffusers | regional_dyn_fa3_vae | `--compile regional --dynamic --attn _flash_3_hub --compile-vae` | c=1: 1.767, c=8: 1.793 | 0.566 | 1.780 | n/a (reference impl.) | yes |
| diffusers | full_dyn | `--compile full --dynamic` | c=1: 1.616, c=8: 1.643 | 0.619 | 1.630 | n/a (reference impl.) |  |
| diffusers | full_dyn_fa3 | `--compile full --dynamic --attn _flash_3_hub` | c=1: 1.629, c=8: 1.669 | 0.614 | 1.649 | n/a (reference impl.) |  |

### Exact server commands (round 1)

```
/root/work/mstar-main/.venv/bin/python -m mstar.cli.main serve flux2_klein_9b --config /root/work/mstar-main/configs/flux2_klein_9b.yaml --gpus 0 --port 8400
/root/work/mstar/.venv/bin/python -m mstar.cli.main serve flux2_klein_9b --config /root/work/mstar/configs/flux2_klein_9b.yaml --gpus 0 --port 8400
/root/work/engines/sgld/bin/sglang serve --model-path black-forest-labs/FLUX.2-klein-9B --num-gpus 1 --host 127.0.0.1 --port 8400 --scheduler-port 8500 --performance-mode speed --enable-torch-compile true --batching-max-size 8
/root/work/engines/vllmomni/bin/vllm serve black-forest-labs/FLUX.2-klein-9B --omni --host 127.0.0.1 --port 8400 --diffusion-compile-granularity full
```

Versions: diffusers: {"packages": ["diffusers==0.41.0", "kernels==0.17.2", "torch==2.14.1+cu132", "transformers==5.20.2", "triton==3.8.0"]}; mstar-main: {"imports": "/root/work/mstar-main/mstar/__init__.py 2.12.1+cu132 0.41.0", "commit": "HEAD abdfd580296a843f62ce967ddb93b0a3dfb225e3"}; mstar-pr: {"imports": "/root/work/mstar/mstar/__init__.py 2.12.1+cu132 0.41.0", "commit": "perf/overnight-all d248477ec795c6ddf0716e23932be15be2b573e2"}; sglang: {"packages": ["cache-dit==1.5.1", "diffusers==0.37.0", "flash-attn-4==4.0.0b36", "flashinfer-python==0.7.1", "kernels==0.14.1", "kernels-data==0.16.2", "sgl-deep-ep==0.1.2.post1", "sgl-deep-gemm==0.2.2", "sglang==0.5.22", "sglang-kernel==0.4.9", "torch==2.14.1+cu132", "transformers==5.19.0", "triton; vllmomni: {"packages": ["cache-dit==1.5.0", "diffusers==0.40.0", "fa3-fwd==0.0.3", "flashinfer-python==0.6.18.post1", "kernels==0.16.1", "kernels-data==0.16.2", "torch==2.13.0+cu132", "transformers==5.14.1", "triton==3.7.1", "vllm==0.30.0", "vllm-omni==0.30.0"]}

### Timed results (3 rounds, fresh server per round, rotated order)

Arm configs: M* PR = config; M* main = config; vllm-omni = full; SGLang = compile_b8; diffusers (tuned, offline batch) = regional_dyn_fa3_vae

## Throughput (img/s), steps=4: mean [min-max] over rounds; n rounds; failed

| arm | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* PR | 1.560 [1.557-1.564] n=3 | 1.719 [1.714-1.727] n=3 | 1.740 [1.736-1.744] n=3 | 1.748 [1.746-1.749] n=3 | 1.750 [1.749-1.751] n=3 |
| M* main | 1.537 [1.531-1.542] n=3 | 1.694 [1.693-1.694] n=3 | 1.721 [1.720-1.722] n=3 | 1.729 [1.729-1.731] n=3 | 1.733 [1.732-1.733] n=3 |
| vllm-omni | 1.354 [1.351-1.357] n=3 | 1.564 [1.563-1.565] n=3 | 1.568 [1.560-1.576] n=3 | 1.571 [1.570-1.572] n=3 | 1.569 [1.562-1.573] n=3 |
| SGLang | 1.719 [1.717-1.720] n=3 | 1.865 [1.863-1.865] n=3 | 1.816 [1.815-1.816] n=3 | 1.845 [1.843-1.846] n=3 | 1.856 [1.855-1.858] n=3 |
| diffusers (tuned, offline batch) | 1.762 [1.759-1.767] n=3 | 1.765 [1.764-1.766] n=3 | 1.775 [1.775-1.776] n=3 | 1.781 [1.779-1.782] n=3 | 1.777 [1.775-1.779] n=3 |

M* PR throughput relative to each arm (mean / mean):

| vs | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---:|---:|---:|---:|---:|
| M* main | 1.02x | 1.01x | 1.01x | 1.01x | 1.01x |
| vllm-omni | 1.15x | 1.10x | 1.11x | 1.11x | 1.12x |
| SGLang | 0.91x | 0.92x | 0.96x | 0.95x | 0.94x |
| diffusers (tuned, offline batch) | 0.89x | 0.97x | 0.98x | 0.98x | 0.99x |

## B=1 latency (s), steps=4, c=1: mean of round means [min-max]; p50; p99 (per round, averaged); requests per round

| arm | mean | p50 | p99 | n req/round | rounds | failed |
|---|---:|---:|---:|---:|---:|---:|
| M* PR | 0.611 [0.610-0.612] | 0.612 | 0.627 | 32 | 3 | 0 |
| M* main | 0.620 [0.619-0.621] | 0.620 | 0.634 | 32 | 3 | 0 |
| vllm-omni | 0.715 [0.714-0.716] | 0.715 | 0.727 | 32 | 3 | 0 |
| SGLang | 0.570 [0.570-0.571] | 0.571 | 0.578 | 32 | 3 | 0 |
| diffusers (tuned, offline batch) | 0.567 [0.566-0.569] | 0.567 | 0.579 | 32 | 3 | 0 |

1-min load average at cell start: median 2.4, max 5.8 over 75 cells (192 vCPUs).

### Profile (phases_c1, MSTAR_PHASE_TIMING=100, benchmark/worker_phases): GPU-bound (worker waits on the GPU most of each step); event_sync / iter_total up to 105%

<details><summary>phase tables</summary>

```
=== worker_0  iters 100-200  records=2  batch=1.00 (range 1.0-1.0) ===
    work: 200 request-steps (bs x iters) -- us/1k is cost per unit of output, comparable across runs with a different iteration mix
phase                                      mean_ms   p50_ms   p95_ms  samples      us/1k
----------------------------------------------------------------------------------------
await_gpu                                    0.839    0.000    4.870      199  834850.0 
engine.collect_outputs                       0.030    0.030    0.040      172   25800.0 
engine.declare_step                          0.000    0.000    0.000      172       0.0 
engine.forward                               1.192    0.360    4.900      172 1025350.0 
engine.postprocess                           0.000    0.000    0.010      200       0.0 
engine.preprocess                            0.085    0.085    0.150      172   73050.0 
gpu_exec                                    78.589  121.555  129.470      19877803100.0 
iter_total                                  73.690  126.980  130.180      20073690000.0 
rust.send_outputs.call                       0.020    0.025    0.050      171   17100.0 
rust.send_outputs.marshal                    0.030    0.020    0.080      171   25650.0 
speculate                                    0.095    0.085    0.180      114   54100.0 
submit_spec                                  0.325    0.360    0.570      142  230600.0 
worker.await_admit_settled                   3.745    3.730    7.540       58 1086050.0 
worker.gpu_thread.await_plan                 0.010    0.000    0.030      142    7100.0 
worker.gpu_thread.exec                       1.218    0.560    5.090      200 1217900.0 
worker.gpu_thread.prepare_inputs             1.045    0.060    6.940      200 1045050.0 
worker.postprocess.check_stop                0.245    0.165    0.390      171  209200.0 
worker.postprocess.check_stop_tail           0.010    0.010    0.020      171    8550.0 
worker.postprocess.cleanup_inputs            0.010    0.010    0.020      199    9950.0 
worker.postprocess.completion_event_sync    90.081  124.040  128.810      17177019650.0 
worker.postprocess.event_sync               77.400  120.355  128.800      19977013100.0 
worker.postprocess.pending_loop_stops        0.010    0.010    0.010      171    8550.0 
worker.postprocess.prematerialize            0.235    0.155    0.380      171  200650.0 
worker.postprocess.register_outputs          0.065    0.000    0.560      171   55500.0 
worker.postprocess.route                     0.035    0.030    0.060      171   29900.0 
worker.postprocess.route_outputs             0.105    0.095    0.180      171   89750.0 
worker.postprocess.send                      0.055    0.050    0.140      171   47000.0 
worker.postprocess.send_outputs              0.075    0.070    0.160      171   64100.0 
worker.postprocess.send_prep                 0.010    0.010    0.010      171    8550.0 
worker.postprocess.stop_loops                0.010    0.000    0.030      171    8550.0 
worker.postprocess.store_tensors             0.060    0.055    0.100      171   51300.0 
worker.postprocess.update_lru                0.000    0.000    0.000      171       0.0 
worker.postprocess_batch                    77.865  120.740  129.150      19977475400.0 
worker.schedule                              0.012    0.005    0.100      939   56550.0*
worker.schedule_yield_away                   0.030    0.010    0.100       85   12750.0 
worker.submit_spec.launch_wait               0.285    0.335    0.550      142  202200.0 
    * sampled on passes that ran no batch, so its time is spread over work it did not all do: us/1k is an upper bound for this row
```

</details>

### Summary (generated from the tables above)

- vs M* main: throughput 1.01x (c=16) to 1.02x (c=1); B=1 latency 0.611 s vs 0.620 s (0.99x). M* PR is not slower at any point.
- vs vllm-omni: throughput 1.10x (c=2) to 1.15x (c=1); B=1 latency 0.611 s vs 0.715 s (0.85x). M* PR is not slower at any point.
- vs SGLang: throughput 0.91x (c=1) to 0.96x (c=4); B=1 latency 0.611 s vs 0.570 s (1.07x). M* PR is slower at c=1 (0.91x), c=2 (0.92x), c=4 (0.96x), c=8 (0.95x), c=16 (0.94x).
- vs diffusers (offline batch, no serving): throughput 0.89x (c=1) to 0.99x (c=16); B=1 latency 0.611 s vs 0.567 s (1.08x). M* PR is slower at c=1 (0.89x), c=2 (0.97x), c=4 (0.98x), c=8 (0.98x), c=16 (0.99x).

<details><summary>notes.txt (startup times, health checks, decisions)</summary>

```
notes for klein9b (2026-10-23 14:34 UTC)

server startup seconds per arm (all starts, incl. cold compiles): mstar-pr: min 42 median 42 max 75 n=6; mstar-main: min 60 median 65 max 102 n=4; vllmomni: min 21 median 48 max 87 n=7; sglang: min 36 median 69 max 117 n=8

health (per server log):
  main_mstar-main_config_r1: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-main_config_r2: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-main_config_r3: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r1: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r2: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_mstar-pr_config_r3: {"failed_capture": 0, "graph_runtime_rust": true, "recompile_lines": 0, "errors": 0}
  main_sglang_compile_b8_r1: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-23 13:46:46] Configuring layerwise offload for Flux2Transformer2DModel: transforme"], "errors": 0}
  main_sglang_compile_b8_r2: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-23 14:03:05] Configuring layerwise offload for Flux2Transformer2DModel: transforme"], "errors": 0}
  main_sglang_compile_b8_r3: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-23 14:18:45] Configuring layerwise offload for Flux2Transformer2DModel: transforme"], "errors": 0}
  main_vllmomni_full_r1: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  main_vllmomni_full_r2: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  main_vllmomni_full_r3: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  tune_base_sglang_base_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_base_vllmomni_base_r0: {"attention": ["FLASH_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_bcg_b8_sglang_bcg_b8_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_bcg_sglang_bcg_r0: {"attention": ["fa"], "speed_mode_msgs": ["performance_mode=speed keeps torch.compile disabled for this model (pass --enable-t"], "offload_msgs": [], "errors": 0}
  tune_compile_b8_sglang_compile_b8_r0: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-23 13:02:28] Configuring layerwise offload for Flux2Transformer2DModel: transforme"], "errors": 0}
  tune_compile_sglang_compile_r0: {"attention": ["fa"], "speed_mode_msgs": [], "offload_msgs": ["[10-23 12:55:56] Configuring layerwise offload for Flux2Transformer2DModel: transforme"], "errors": 0}
  tune_cudnn_vllmomni_cudnn_r0: {"attention": ["CUDNN_ATTN"], "regional_compile": true, "enforce_eager": false, "errors": 0}
  tune_eager_vllmomni_eager_r0: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}
  tune_full_vllmomni_full_r0: {"attention": ["FLASH_ATTN"], "regional_compile": false, "enforce_eager": false, "errors": 0}

decisions:
  2026-10-23 13:24:34 tuned_config: {"engine": "diffusers", "config": "regional_dyn_fa3_vae", "candidates": [[1.7801666661992679, "regional_dyn_fa3_vae"], [1.6543828841908845, "regional_dyn_fa3"], [1.6487392934824232, "full_dyn_fa3"], [1.6295093049048965, "full_dyn"], [1.6248909710381334, "regional_dyn"], [1.6236889374912653, "regiona
  2026-10-23 13:33:31 gate: {"verdicts": {"mstar_pr": "PASS", "mstar_pr_b": "PASS", "mstar_main": "PASS", "sglang__base": "PASS", "sglang__bcg": "PASS", "sglang__bcg_b8": "PASS", "sglang__compile": "PASS", "sglang__compile_b8": "PASS", "vllmomni__base": "PASS", "vllmomni__cudnn": "PASS", "vllmomni__eager": "PASS", "vllmomni__f
  2026-10-23 13:33:31 tuned_config: {"engine": "vllmomni", "config": "full", "gate_pass": true, "candidates": [[1.460405480464895, "full", "PASS"], [1.4450468836437518, "base", "PASS"], [1.4360967698269846, "cudnn", "PASS"], [1.3593541631954291, "eager", "PASS"]], "rule": "best tune score among gate-passing knob sets"}
  2026-10-23 13:33:31 tuned_config: {"engine": "sglang", "config": "compile_b8", "gate_pass": true, "candidates": [[1.7781264737173148, "compile_b8", "PASS"], [1.7539606523402222, "bcg_b8", "PASS"], [1.7520298821271554, "compile", "PASS"], [1.735731476533582, "base", "PASS"], [1.7335678240722798, "bcg", "PASS"]], "rule": "best tune sc
  2026-10-23 14:33:19 profile_points: {"points": [1], "ratios": {"1": 0.8853013945916687, "2": 0.9219329480599454, "4": 0.9584011130709397, "8": 0.9473066671111461, "16": 0.942853615130029}, "rule": "point where mstar-pr / best competitor is lowest, plus c=1 if mstar-pr trails there"}
```

</details>

## Caveats

- **diffusers rows are not served.** They are in-process offline batches (b = c), timing the pipeline call
  only: no HTTP, no queueing, no PNG/mp4 encode. M*, vllm-omni and SGLang latencies include HTTP, image/mp4
  encoding and base64 on the response; their c > 1 rows are closed-loop serving with c requests in flight.
- **Startup is excluded.** Cold compiles (M* Z-Image ~28 min on the first start, ~4 min warm) and capture
  happen before the untimed warm pass; every timed cell follows one untimed pass over the whole prompt set.
- **Prompt lengths vary** (PartiPrompts range from one word to several sentences). M*'s Z-Image config captures
  caption-length buckets 32/64/96/128 tokens and batches only equal buckets; other lengths run its eager batched
  path. This is the realistic case the overnight tuning (fixed prompts) did not exercise.
- **Video noise on vllm-omni** comes from a CUDA generator (its video form takes no per-request
  `generator_device`), so its Wan2.2 outputs cannot be pixel-compared with the CPU-generator reference; its video
  gate is distribution-level only. SGLang takes `generator_device: cpu` for both images and video.
- **SGLang's video API is asynchronous** (POST, then poll every 0.1 s, then fetch the mp4), which adds up to
  ~0.1 s per video to its latency.
- **M* wan22 servers run with the HF hub online** (its tokenizer load probes a config.json that the cache does not
  hold and fails under `HF_HUB_OFFLINE=1`); every other server runs offline. This affects only the first request's
  tokenizer load, before any timed cell.
- **Each engine runs its own pinned torch** (M* 2.12.1, vllm-omni 2.13.0, SGLang and diffusers 2.14.1).
- **p99 at B=1** is over 32 requests per round for images (close to the maximum) and 3 for video; read it as a
  tail indicator, not a percentile estimate.
- **Few video requests per point** (3-16 per round): the DiT is GPU-bound and serial, so the spread is small, but
  the min-max over rounds is shown for every point.
- **Competitor tuning** is one short sweep with a single process per knob set; differences between knob sets near
  the cross-process noise make the pick "best observed".

## Not covered / failed

- 2026-10-22 04:50:40 zimage: pass_failed {"tag": "tune_base", "arm": "vllmomni", "knob": "base", "round": 0}
- 2026-10-22 04:51:35 zimage: pass_failed {"tag": "tune_cudnn", "arm": "vllmomni", "knob": "cudnn", "round": 0}
- 2026-10-22 04:52:38 zimage: pass_failed {"tag": "tune_full", "arm": "vllmomni", "knob": "full", "round": 0}
- 2026-10-22 05:00:26 zimage: pass_failed {"tag": "tune_base", "arm": "vllmomni", "knob": "base", "round": 0}
- 2026-10-22 20:42:54 fish_s2pro: pass_failed {"tag": "collect_mstar_main", "arm": "mstar-main", "knob": "python", "round": 0}
- 2026-10-22 21:03:54 fish_s2pro: pass_failed {"tag": "main", "arm": "sgomni", "knob": "mrr128", "round": 1}
- 2026-10-23 02:32:26 chatterbox: pass_failed {"tag": "tune_graphs", "arm": "cbvllm", "knob": "graphs", "round": 0}
- 2026-10-23 04:21:51 higgs_audio: pass_failed {"tag": "tune_sdpa", "arm": "higgs_official", "knob": "sdpa", "round": 0}
- 2026-10-23 10:51:11 bagel_2gpu: pass_failed {"tag": "tune_cfgp2", "arm": "vllmomni", "knob": "cfgp2", "round": 0}
- 2026-10-23 11:55:53 bagel_2gpu: pass_failed {"tag": "main", "arm": "vllmomni", "knob": "cfgp2", "round": 1}
- 2026-10-23 12:21:09 bagel_2gpu: pass_failed {"tag": "main", "arm": "vllmomni", "knob": "cfgp2", "round": 2}
- Multi-GPU rows (Wan2.2 CFG split on 2 GPUs, DP2) were not run: out of the night's budget.

