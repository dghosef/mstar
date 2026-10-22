# world: Kokoro and the held-output fix (35462fc11). No regression reproduced, no code change

**Verdict: keep 35462fc11 as is.** The proposed mechanism was checked and is wrong. The fix's call costs nothing measurable on Kokoro, so there is nothing to change.

## Proposed mechanism, checked against the code path and the run
- The hypothesis was that Kokoro outputs are sent a step after they're produced and now pay a D2H plus a sync. Kokoro does not do that.
- Kokoro's graph is one node in `chunk_loop`. Its `audio_chunk` goes to EMIT_TO_CLIENT on every iteration. `_register_outputs` stages it in the same `_postprocess_batch`, and that happens before `release_host_copies` runs.
- Instrumented arm (head plus a counter in `SharedMemoryCommunicationManager._send_source`, which `ArenaShmCommunicationManager._stage_one` also uses). Worker sends over a full Kokoro c=32 regress round: `host=100 device=0` (shape (115200,), cuda:0). Every send read the stored host copy, and none fell back to the device tensor.
  - Run: `results/world/kokoro_dbg2/kokoro/base_r0/server.log`, the DBGSEND lines.

## Phase profile (MSTAR_PHASE_TIMING, regress.py profile pass with 15x the load, matched batch size 8.5-9)
| phase (mean ms) | revert | head |
| --- | --- | --- |
| iter_total | 34.18 | 33.92 |
| event_sync | 15.63 | 14.71 |
| register_outputs (mean / p50) | 1.03 / 0.76 | 0.64 / 0.83 |
| send_outputs | 0.24 | 0.25 |
| postprocess_batch | 17.39 | 16.10 |

- No phase grew. Results: `results/world/kokoro_prof_long/`.

## A/Bs (regress.py cell kokoro, 4 rounds ABBA, H200 GPU0, host load 5-16)
- revert (head with 35462fc11's code reverted) vs head, on ovn-artts:
  - c=32: 793/786/685/760 vs 765/766/764/786 audio-s/s, head +1.9% (median 773 vs 765).
  - c=32 TTFA: head -2.1%.
  - c=1 TTFA: head +2.1%.
  - WER: identical, 0.94%.
- noop (head without the `release_host_copies` call) vs head, on ovn-omni:
  - c=32: 793/779/809/740 vs 783/773/811/793, head +1.3%.
  - c=32 TTFA: head -1.1%.
- Both are inside this host's ±4-6% round-to-round spread.
  - Note: ovn-omni, ovn-artts and ovn-world are containers on one physical host (3.135.136.10), so CPU-bound cells also share CPU with whatever else runs there.
- I could not reproduce the integrator's -6.9%. My hypothesis, not checked: host noise, since base r2 here dropped to 685 at load 8.

## Correctness
- `test_shm_tensor_comm.py::test_a_held_output_is_sent_from_its_own_tensor_not_the_reused_host_copy` passes on head and fails on revert (run on ovn-artts).
- V-JEPA2-AC correctness is unchanged, since no code changed:
  - post-fix waves of 4 concurrent identical requests are bit-identical across rows (`results/world/vjepa2_ac_batch_bug.md`);
  - c=1 outputs are byte-identical between perf/ovn-world and shared 387ce372c (`results/world/vjepa2ac_final_bytes`).

## Suggestion
- If the -5.5% shows up again in a combined A/B, rerun the two-arm comparison above alone on a quiet host (nothing else on 3.135.136.10).
- A real cost from this fix would have to come from some other model that sends held outputs. Today that is only the V-JEPA2 rollouts' accumulated outputs, and paying a D2H there is the point of the fix.
