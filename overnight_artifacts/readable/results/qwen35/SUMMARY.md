# qwen35 — overnight perf push (final, 2026-10-21 ~12:15 PDT)

Box ovn-qwen35, 2x H200 (Eras), branch `perf/ovn-qwen35` (worktree `/Users/josephtan/mstar-ovn-qwen35`),
base = `perf/overnight-shared`. Harness: `benchmark/ovn_qwen35/` (untracked): `ab.py` (arm A/B driver over
`benchmark.runner` closed loop, text dataset, ignore_eos, 256 output tokens, fresh server per arm,
alternating order), `spy.py`/`diag.py`/`threadcpu.py` (py-spy, top, per-thread /proc CPU),
`kv_micro.py`/`rec_micro.py` (CPU microbenchmarks of resource admit/plan/commit), hooks/sitecustomize.py
(per-process switch interval, yappi, torch-profiler kernel table, A/B toggles). Results: this directory.

## Qwen3.5-0.8B text (configs/qwen3_5_0.8b.yaml, 1 GPU) — CPU-bound

Baseline (shared base 55b0215a3, H200): c=1 ~680 tok/s, c=32 ~13.3k, c=64 ~15.3k (runner, 256 tokens).

Where the time goes (profile pass MSTAR_PHASE_TIMING=100; py-spy; /proc per-thread CPU):
- c=1 and c=32: CPU-bound (event_sync 0.03-0.1 ms of a 1.4 / 2.6 ms step). The step's critical path is
  host-only: commit(N) -> plan-thread pre_plan(N+1) -> GPU-thread promote/preprocess/launch(N+1) ->
  commit(N+1). The worker process is GIL-bound at c=32: main + gpu + plan threads sum to 1.07 cores, so
  any Python removed from any of its threads is step time.
- c=64 fixed-length closed loop is capped by the API server process, not the worker: all 64 requests
  finish together and the worker idles ~250 ms per wave (utilisation 0.6 in those windows, 1.0
  otherwise, from phase-log timestamps). The API server's three threads sum to 1.0-1.16 cores at
  c=32/64 (GIL-saturated). benchmark.runner is not the cap (a 4-process light client gives the same).

Kept:
1. `MSTAR_INGRAPH_DECODE_ROPE` default on (0cb096086, model-local): rotary tables built inside the
   captured decode graph instead of eagerly on the GPU thread's critical path (engine.preprocess
   0.20 -> 0.03 ms). Same build, env arms, 2 rounds: c=32 13487 -> 15305 (+13%), c=1 679 -> 775 (+14%),
   c=64 flat (API-capped). 9B: neutral (GPU-bound). Correctness: greedy differs from the staged tables
   only at near-tie tokens (base vs base across processes 8/8 identical; rope vs base 2/8, divergences
   100-800 chars in, synonym-level); eager vs compiled tables bit-identical over positions 0-32767.
2. `api_switch_interval_s: 0.001` (d26d96909): per-deployment GIL switch interval of the API server
   process, set in the Qwen3.5 configs; unset keeps 5 ms for every other model. MERGED into
   perf/overnight-shared as aa2cfa4fa (integrator: c=64 +10.9% over 4/4 rounds, c=32 +3.5%). 3 rounds: c=64
   15716 -> 16994 (+8.1%), c=32 +7% (noisy), c=1 +3.8%; integrator 4 rounds: c=64 +3.7%, c=32 +3.0%.
   Control (same interval in worker processes only): noise. Opt-in because the integrator measured
   Kokoro -3.6% at c=32 with it global. Proposal: shared/proposals/qwen35-api-switch-interval.md.

Leads (measured, not defaults):
- Rust HTTP frontend (`--rust-frontend`, opt-in; `cargo build --release` in rust/server): same build,
  2 rounds: c=64 16534 -> 23972 tok/s (+45%), c=32 14381 -> 16209 (+13%), c=1 +5%; greedy 8/8 identical.
  The Python API server is the high-concurrency cap for small streaming models.

Dropped (numbers in NOTES.md): MSTAR_SAMPLER_INGRAPH_SCATTER, MSTAR_KV_PLAN_CACHE (noise, twice, incl.
under the Rust frontend); data-worker multi-request preprocess pass (TTFT unchanged); switch interval in
every process (c=1 -10%) or in workers only (noise).

## Qwen3.5-9B (configs/qwen3_5_9b.yaml, 1 GPU) — GPU-bound

Baseline: c=1 208 tok/s, c=16 2.79k, c=64 8.1k; c=64 event_sync 4.6 of 7.2 ms/step. Kernels (torch
profiler, 200 replays at c=64): GEMMs ~55% (near weight-bandwidth roofline at bs~61), GDN decode kernel
20% (fp32 recurrent state, ~2.9 GB/step). In-graph rope neutral.
Lead: `gdn_state.state_dtype: bfloat16` -> c=64 8122 -> 8985 (+10.6%), c=16 +4.7%, c=1 +1.6%; greedy 3/8
identical vs fp32 (each arm self-consistent 8/8): a numerics change that needs a task-accuracy gate.

## Qwen3.5-9B TP2 (configs/qwen3_5_9b_tp2.yaml, 2x H200)

Baseline: c=1 243 tok/s, c=16 2.90k, c=64 7.44k — slower than one GPU at c=64 (8.12k). Profile at
c=16: leader step 5.4 ms with event_sync 3.7 ms; the serial TP protocol (MSTAR_TP_ASYNC_SCHED default 0)
schedules and plans each step between the ranks' forwards (engine.plan.fresh 0.63 ms every step).

Kept:
3. `tp_async_sched: true` in configs/qwen3_5_9b_tp2.yaml (91a4c505c, config only): c=1 243 -> 336
   (+38%), c=16 2895 -> 4181 (+44%), c=64 7441 -> 11506 (+55%), 2 rounds. Leader step 5.4 -> 3.6 ms,
   now GPU-bound (event_sync 2.1 ms). Greedy 8/8 identical at c=1; c=8 identical to base's c=8.
   Stress (420 s, 48 lanes, EOS on/off, 1-600 tokens, long prompts, 20% mid-stream client aborts):
   8423 completed / 1660 aborted / 0 errors, healthy after; serial path on the same harness 3735/768/0.
   Re-validated with the config value (not the env var), tree = branch on shared d1c91505a:
   - stress v2, 360 s same mix: 7126 completed / 1409 aborted / 0 errors; 4 cycles of 64 long streams
     (all finished before the cancel point); then mass cancel: 6 cycles x 64 streams dropped at once
     4 s into 2000-token generations (384 cancelled mid-stream); server healthy, a fresh request
     completes after each run; no worker traceback (only the known SIGINT-shutdown one in both arms).
   - greedy vs the serial protocol (MSTAR_TP_ASYNC_SCHED=0): 8/8 identical at c=1, 7/8 at c=8 (batch
     composition differs run to run; serial c=1 vs its own repeat is 8/8).

## Qwen3.5-27B TP4 (configs/qwen3_5_27b_tp4.yaml, 4x H200, guest on ovn-regress)

5. `tp_async_sched: true` in configs/qwen3_5_27b_tp4.yaml (same mechanism as 3): 1 round (cold
   compile is 21 min per new tree): c=1 156.5 -> 190.6 tok/s (+22%), c=16 1866 -> 2362 (+27%), c=64
   5196 -> 7038 (+35%). Greedy 8/8 identical to the serial protocol at c=1 and at c=8.

## Qwen3.5-4B / 2B text (1 GPU each)

Same harness; base = rope off + no switch-interval key; kept = branch defaults; rustfe = kept + --rust-frontend.
| model | c | base | kept | rustfe |
| --- | --- | --- | --- | --- |
| 4B | 1 | 328 | 331 (+1%) | 332 |
| 4B | 32 | 6604 | 6732 (+1.9%) | 6806 (+3.1%) |
| 4B | 64 | 10040 | 10211 (+1.7%) | 10609 (+5.7%) |
| 2B | 1 | 627 | 637 (+1.7%) | 634 |
| 2B | 32 | 12277 | 12436 (+1.3%) | 12736 (+3.7%) |
| 2B | 64 | 15709 | 16508 (+5.1%) | 20609 (+31%) |

4B is mostly GPU-bound (c=32 event_sync 2.1-2.6 of 4.4 ms/step). Kernels at c=32 (200 replays):
GEMMs ~62% (the nvjet/Triton-template GEMMs at M=32 run at ~60% of weight-bandwidth roofline), GDN
decode 18%, fused add-RMSNorm 4.4%. 2B sits between: at c=64 the API server is the cap again.

Kept:
4. Decode preprocess routed-row index through pinned memory (f044a984e, model-local): a request's first
   decode step after prefill built its index with a pageable torch.tensor(..., device=cuda), a
   synchronous H2D. 4B c=32 profile (kept vs kept+fix): engine.preprocess.loopback mean 0.203 ->
   0.033 ms, p95 4.71 -> 0.13 ms; gpu_thread.exec 0.73 -> 0.57 ms. Throughput flat on GPU-bound 4B
   (2 rounds); greedy 8/8 identical at c=1.

## Branch and commits (perf/ovn-qwen35, on perf/overnight-shared 387ce372c)

- 464820337 [Qwen3.5][Perf] Build captured decode rotary tables in the graph by default
- (merged to shared as aa2cfa4fa) api_switch_interval_s per-deployment, 1 ms for the Qwen3.5 configs
- 15af554d6 [Qwen3.5][Perf] 9B TP2: tp_async_sched: true
- f044a984e [Qwen3.5][Perf] Decode preprocess: routed rows' index through pinned memory (sync removed;
  0.8B 3 rounds: c=1 +6%, c=32 +1%, c=64 +1.6%, all within noise; 4B flat)
- 08828d5c8 [Qwen3.5][Perf] 27B TP4: tp_async_sched: true
Dropped from the branch: NDJSON one-write-per-batch (c=32 +5%, c=64 +1%, noisy) and a KV admit
micro-optimisation (34 -> 27 us/step in a CPU microbenchmark, invisible end to end).

## CPU/GPU-bound verdicts

- 0.8B: CPU-bound at every concurrency before and after; worker GIL-bound at c=32 (1.07 cores across
  its three threads); at c=64 the Python API server is the cap (1.0-1.16 cores).
- 2B: CPU-bound at c=64 (API server), mixed at c=32. 4B / 9B: GPU-bound (GEMMs near or at ~60% of the
  weight-bandwidth roofline, fp32 GDN state traffic 18-20%).
- 9B TP2 / 27B TP4: host-bound by the serial TP protocol before; GPU-bound after tp_async_sched.

## Open leads

1. Rust HTTP frontend as the default for streaming text models: +45% at c=64 on 0.8B, +31% on 2B.
2. `tp_async_sched: true` for the other TP/SP configs (Omni thinker_tp2, Cosmos super_tp2, ...): the
   same serial-protocol cost should be there; the code path held up under abort / mass-cancel stress.
3. bf16 GDN recurrent state for 9B (+10.6% at c=64): numerics change, needs a task-accuracy gate.
4. 0.8B worker GIL: only structural changes help now (pre-plan N+1 before commit N from predicted
   lengths, or move steady-decode planning out of Python); per-row micro-opts measure as noise.
5. Per-token API-server work (InlineToken / queue / ResultChunk per token across three threads) could
   be batched per frame for the Python frontend.

Harness and evidence: benchmark/ovn_qwen35/ in the worktree (untracked); per-run results in this
directory (bench.jsonl, summary.md, correct.txt, profile/*.phases.txt, tprof*/, tcpu08/, yappi08/,
tp2_stress*/, guest_regress/ (27B), guest_qwen35vl/ (0.8B routed A/B #1)). All servers stopped;
ovn-qwen35, ovn-qwen35vl and ovn-regress GPUs idle.
