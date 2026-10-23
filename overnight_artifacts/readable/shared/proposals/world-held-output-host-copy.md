# world: BUG FIX — held outputs were sent from the step's reused pinned host copy

- Branch `perf/ovn-world`, commit `4fcec69e9` (rebased on d1c91505a; was bcb88d101) (files: mstar/worker/worker.py, mstar/communication/tensors.py,
  mstar/communication/tensor_store.py, test/modular/test_shm_tensor_comm.py). Correctness fix, not perf.
- Write-up with repro + evidence: `~/runs/overnight/results/world/vjepa2_ac_batch_bug.md`.
- Mechanism: `_prematerialize_for_check_stop` copies every CUDA output into pinned buffers reused next step;
  the store keeps them as host copies for SHM/arena sends (`_send_source`). Outputs sent at a LATER step
  (Loop.accumulated_outputs; any edge held past the step) serialized the buffer's later contents. V-JEPA2
  rollouts returned the last iteration H times; under batching, a timing-dependent one (3-31% rel L2 drift).
- Fix: after `_register_outputs`, `tensor_manager.release_host_copies(stored.flat_uuids)` drops this step's
  host copies; later sends read the stored device tensor. Cost: one dict pop per stored tensor per step.
- Tests: new test fails without / passes with; test_shm_tensor_comm 46 passed. Full CPU Core run on the box in
  progress (results appended to NOTES).
- Affects: any model whose output edge is sent after its producing step on SHM/arena transports. Only
  vjepa2 declares accumulated_outputs today; persisted edges (persist=True) to remote consumers may also
  be affected — worth a regression pass on streaming TTS/Omni (their outputs are sent same-step, so no
  change expected) and a CPU-overhead check at high-concurrency decode (Qwen3.5 0.8B c=64).

## Verdict (integrator 02:43): MERGED as 35462fc11 + 7fd1e3589 (test-fake fix added by integrator)

**(1) Async worker ordering** (read .claude/skills/async-worker/SKILL.md and traced postprocess on the main thread):
- `release_host_copies(stored.flat_uuids)` runs right after `_register_outputs`. That is where `register_for_send_uuids` synchronously stages every remote-bound tensor of this step into the arena or an SHM file (`_stage_one` -> `_send_source`). So the copies are released after this step's serialization.
- `_stream_and_send`, which runs after the release, reads no host copies: local streaming takes `get_tensor` (device) and clones, and inline tokens come from `cpu_outputs`. The same step loses nothing.
- Speculative N+1: its prematerialize D2H into the reused pinned buffers happens in N+1's postprocess (step 7). That runs on the same main thread after N's postprocess has returned, so it can't overlap N's staging, and the release doesn't move it.
- A send still in flight on another thread: `_stage_one` has a concurrent-duplicate guard, so other threads can register. One that took a host-copy reference before the release still holds that tensor; one that calls `_send_source` after it falls back to the device tensor. The device tensor is stable, because accelerator-graph outputs are `.clone()`d out of the static buffers (accelerator_graph_runner.py:1566/1572). So the fix narrows the window; it doesn't open one. The dict pop is GIL-atomic. TP: no collective and no rank-dependent control flow (invariant 11).
**(2) Invariant 5 / Rust:** host copies live only in the Python `TensorStore._tensors_cpu`. The Rust SHM arena gets its bytes from that same `_stage_one` (Python `host.copy_` into the arena segment view), and `rust/src` keeps no host copies (grep). MSTAR_RUST_ZMQ carries control frames, not tensor bytes. So both sides run the fixed path. test_shm_tensor_comm + test_row_output_store: head 46/49 pass (Rust off/on), cand 47/50 (Rust off/on, including the new test). pytest test/rust/ on cand: 359 passed. The new test, run against the unfixed head, FAILS (checked).
**(3) Overhead and regressions** (base = shared d1c91505a, H200, ABBA, fresh servers):
- Qwen3.5-0.8B t2t, 4 rounds, run twice. Run 1: c64 +3.6%, c32 -8.3%, c1 -7.7% (cand r0/r1 overlapped a flash-attn build at load 70). Rerun on a quiet host (load 5-27): **c64 -0.5%, c32 -0.1%, c1 +2.5%** (all within noise). Greedy 8/8 identical in both runs.
- Qwen3-TTS c=16 streaming, 3 rounds: 97.1/97.3/93.6 vs 82.2 (under build load)/97.6/102.5 audio-s/s; TTFA c1 -1%.
- Qwen3-Omni t2s (configs/qwen3omni_2gpu.yaml, 2 GPUs), 2 rounds: c=8 51.8/51.4 vs 51.7/51.1 audio-s/s (-0.4%), c=8 TTFA p50 57/50 -> 46/45 ms, c=1 rtf +1.7% (noise). The WER harness is off for Omni (its replies aren't verbatim), but it scores the same on both arms (5.83 vs 5.72).
**Tests:** the CPU Core run had ONE new failure, test_worker_drain::test_a_walk_the_cache_served_whole_is_still_routed: its SimpleNamespace tensor-manager fake had no `release_host_copies` (the stub-by-name trap AGENTS.md warns about). I fixed the fake in 7fd1e3589. worker_drain, micro_scheduler, admit_failure and graph_speculation then pass 156/156. No other new failures.

## Follow-up (integrator 13:25): measurable Kokoro cost, fix stays merged

- Quiet-host bisect on ovn-artts (load 5-9), Kokoro c=32, 4 rounds ABBA: shared head a51da9261 vs the same head with 35462fc11 + 7fd1e3589 reverted: **c=32 audio-s/s 616-756 vs 675-785 (-6.9%), c=32 TTFA p50 +11%, rtf p50 +11.6%**. c=1 unchanged (-1.5% TTFA), WER identical (0.94%). This explains the recurring -3 to -5% Kokoro reading in the combined runs.
- Mechanism (hypothesis, not traced): some Kokoro outputs are registered for send at a later step than the one that produced them. After the release they read the device tensor, so each one pays a D2H plus a stream sync on the main thread instead of reusing the prematerialized host copy.
- The fix stays merged, because it corrects data corruption. **world/tts follow-up:** keep a *private* host copy (clone out of the reused pinned buffer) only for uuids that are still unsent at the release, or release only the uuids this step sent. Either keeps the correctness and drops the extra D2H. Check with `py-spy` on the Kokoro worker at c=32 first, to confirm which send path pays.
