# qwen3tts/base: perf stack + Naomi's host-overhead stack

Worktree `~/mstar-qwen3tts-base`, branch `qwen3tts/base`. CPU-only work, nothing pushed.

- Parent 1: perf stack, dghosef `perf/ovn2-final` 4e3de8eae (#858 -> #859 -> #860 + push 2).
- Parent 2: Naomi's non-draft stack top, `pr861` 4d56897c3 (#849 -> #850 -> #851 -> #852 -> #857 -> #861).
- Both stacks fork from main b47e35b31. The perf stack also contains main up to a456839cd
  (#883: pyo3 0.23 -> 0.26, `python` feature, PyO3 surface moved to `rust/src/bindings.rs`).
- #385 (producer-triggered walks, `consumer_walk`) is NOT in 4e3de8eae, so #861's walk-run
  flush bug (`released` unset in #385's new `pop_chunk` branch) is latent here: it applies only
  once this base is rebased onto a main that has #385.

## Step 1: merge pr861 into 4e3de8eae -> 81cb657b6

12 files conflicted (21 hunks). Per hunk, which side won and why:

| File | Hunk | Kept | Why |
| --- | --- | --- | --- |
| docs/installation.rst | FlashInfer/flash-attn bullets | both | #851's FlashInfer pin note + perf stack's reworded flash-attn line (LingBot-World `reference_numerics`). |
| mstar/api_server/data_worker.py | `_emit_inline_tokens` body | pr861 | #849 batches a frame's inline values into one `postprocess_inline` call; perf stack's side was the old per-item body. The perf stack's reason (a list, not a tensor, per inline value, for GIL cost) moved into `Model.postprocess_inline`'s default, see "Outside the conflicts" below. |
| mstar/api_server/entrypoint.py | `typing` import | pr861 | superset (`NamedTuple`, `Protocol`). |
| mstar/model/orpheus/submodules.py | imports | both | `TokenStop` (perf) + `StopKind`, `StopPolicy` (#857). |
| mstar/model/orpheus/submodules.py | `forward_batched` return | pr861 | identical code; #857's comment mentions the runtime-run stop policies, true after the merge. |
| mstar/model/orpheus/submodules.py | `check_stop` / `get_stop_policies` | **new** | Perf stack deleted the hand-written `check_stop` (derived from `TokenStop`); #857 kept it and added `get_stop_policies`. Kept perf stack's `TokenStop` (no hand-written `check_stop`) and wrote `get_stop_policies` from `self.token_stop`'s fields, so both stop paths read one declaration. |
| mstar/model/qwen3_5/submodules.py | imports | both | `TokenStop` (perf) + `StopKind`, `StopPolicy` (#857); `device_loopback_enabled` dropped (perf moved its use into `ARNodeSubmodule`). |
| mstar/model/qwen3_5/submodules.py | `check_stop` .. `check_stop_batched` / `get_stop_policies` | **new** | As Orpheus: perf stack derives `check_stop`, `check_stop_batched`, `step_is_input_free`, `inline_client_signals` from `TokenStop(walk="decode", loopback_signal=...)`; #857's side re-stated them by hand. Kept the derived ones; `get_stop_policies` built from `token_stop` (`{}` off `token_stop.walk`, as #857's). |
| mstar/worker/worker.py | imports (request_info, all_reduce) | neither | both unused in the merged file (perf replaced `flagged_on_any_rank` with `CommGroup.agree`; #861 removed the `CurrentForwardPassInfo` use). |
| mstar/worker/worker.py | stream_buffer / topology imports | pr861 | the stream code moved to `StreamBufferManager`; `ChunkBatchHold` now imported there. |
| mstar/worker/worker.py | `_LazyPerRid` | both | #852's `_host_column` + perf stack's `_LazyPerRid(Mapping)`. |
| mstar/worker/worker.py | consumer cache / `_stream_holds` init | pr861 + port | #861's `StreamBufferManager`; the perf stack's lockstep rule (a follower ingests unheld) passed in as `unheld_nodes=parallel_nodes - parallel_leader_nodes`. `Worker._build_stream_holds` deleted (now `StreamBufferManager.holds`). |
| mstar/worker/worker.py | stream helpers (`_route_streaming_tensor` .. `_stream_chunks_for`) | pr861 + port | #861 moved them into `StreamBufferManager`. The perf stack's change in them (R1.5: `_pop_streaming_edge` leaves a held chunk, so a speculative build's poll holds too) ported into `StreamBufferManager.pop_edge`. |
| mstar/worker/worker.py | `_poll_stream_buffers` body | pr861 + port | `polled = self.streams.poll(idle)`; the perf stack's `ChunkBatchHold` hold/release loop moved verbatim into `StreamBufferManager.poll(idle)`, after #861's `maybe_ready` skip. `_poll_stream_buffers(idle=...)` signature kept (the main loop passes `idle=pending is None`). |
| rust/src/lib.rs | module list | both | perf/main's `#[cfg(feature = "python")] mod bindings;` + #861's `batch_builder`, gated on `python` too (it uses `graph::spec::Sym`, which needs pyo3). |
| rust/src/lib.rs | PyO3 surface | perf | main's #883 moved it into `bindings.rs`; #861's two new pyclasses (`SpecRowsOut`, `PostprocessOut`) registered there instead. |
| rust/src/tensors.rs | module doc | pr861 | #861's store now holds tensor objects; reworded `allow_threads`/"pyo3 0.23" to `detach` (pyo3 0.26 drains its reference pool when `detach` returns: `SuspendAttach::drop`, pyo3-0.26.0 `src/internal/state.rs:252`). |
| test/modular/test_multi_stream_final_flag.py | stream-poll tests | pr861 + port | #861's `possible()` test kept; the perf stack's two ChunkBatchHold tests rewritten against `StreamBufferManager` (`poll(idle)`, `poll_for`, `holds` with `unheld_nodes`). Same assertions. |
| test/modular/test_qwen3_5_check_stop_batched.py | `step_context` stub | perf | perf stack dropped `check_stop_batched`'s `step_walk` (TokenStop version); the merged engine call doesn't pass it. |
| test/modular/test_rid_handles.py | engine_manager stubs (x2) | both | #850's `engine.publisher_nodes/skip_publish` + perf stack's `add_request(rid, cfgs, info)`. |
| test/modular/test_tp_read_failure_acceptance.py | worker stub | pr861 | `w.streams = StreamBufferManager(...)` replaces `w._stream_holds = {}`. |

### Outside the conflicts (forced by the merge)

- `mstar/model/base.py` `Model.postprocess_inline` default: #849's default built
  `torch.tensor([v])` per value; the perf stack had just made inline values reach
  `postprocess` as a list (every torch call drops and retakes the GIL). Default now passes `[v]`.
  Models with their own `postprocess_inline` (Qwen3.5) are unaffected.
- `mstar/worker/stream_buffers.py`: `ChunkBatchHold` port (`holds`, `poll(idle)`, held check in
  `pop_edge`), as above. `mstar/streaming/stream_buffer.py` docstring now names `pop_edge`.
- Rust: `rust/src/bindings.rs` registers `SpecRowsOut` / `PostprocessOut`; `allow_threads` ->
  `detach` in #861's `graph/runtime/postprocess.rs` (pyo3 0.26 deprecates `allow_threads`) and
  one comment in `graph/runtime.rs`. `cargo check` clean for `--features python`, default
  (extension-module), `--no-default-features`, and `rust/server`.

### Checked for both-sides / defeated-guard hunks

- Stops: exactly one source per model (`TokenStop`), read by both the runtime (`StopPolicy`) and
  the Python fallback (`check_stop*`). No model keeps a hand-written `check_stop` beside it.
  `check_stop_batched` has one signature (perf's, no `step_walk`); no caller passes `step_walk`.
- Stream holds: one hold decision per pass, recorded on the buffer (`held`); every pop path
  (`poll`, `poll_for` via `pop_edge`) honours it; `maybe_ready` skip runs before the hold check,
  and a held buffer stays `maybe_ready` (nothing popped), so it is re-examined next pass.
- Engine: `check_stop_for_batch` is the perf stack's form; #857's `get_stop_policies` and
  `step_finished` (-> `extend_prefix_chains`, the only call site) added beside it, no double call.
- Attention / GDN: perf stack's `kv_config.shard(info.head_shards)` and compiled GDN gates
  (`_masked_rows`, `_chunked_gates`, `_unpadded`) auto-merged with #851's direct-plan staging; the
  fields they read (`_plan_borrowed`, `_accelerator_graph`, `token_mask`) are unchanged by #851.
- Qwen3.5 decode loop-back: #852's `_splice_row_ids` replaced `sampler.overwrite_rows`
  (base code, untouched by the perf stack).

### Behaviour the merge had to choose

1. Stop declaration: perf stack's `TokenStop` is the one declaration; #857's runtime stop policy is
   derived from it for Orpheus and Qwen3.5 only (#857's scope). Other `TokenStop` models (Canary,
   Whisper, GLM-5.2, Command A+, Higgs, Qwen3-ASR, Qwen3-Omni, Qwen3-TTS, Chatterbox) still stop via
   `check_stop`; deriving `get_stop_policies` in `ARNodeSubmodule` from any `TokenStop` without
   `generated_key` / `prefill_walks` would extend #857 to all of them (follow-up, not done).
2. Inline postprocess default: list per value (perf), not a tensor (#849).

### Known bugs carried from Naomi's stack (not fixed; the merge didn't force them)

- #849: a failed relay send (`entrypoint._relay_chunks` -> `rust_frontend.relay`) drops every
  relayed request's chunks for that pass silently (pr849 review P2-1).
- #850: every decode step releases the GIL while holding `&mut self` on the runtime pyclass, so
  another thread calling the runtime then gets "Already mutably borrowed" (pr850 review #4).
- #861: walk-run flush leaves `StreamChunk.released` unset in #385's `pop_chunk` branch, and
  `maybe_ready` can starve a walk-gated stream: latent here (no #385 in 4e3de8eae); bites at the
  rebase onto a main with #385.
