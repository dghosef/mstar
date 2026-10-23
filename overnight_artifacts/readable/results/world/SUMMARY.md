# world: summary (V-JEPA2, V-JEPA2-AC, LingBot-World, Waypoint, Cosmos3 DROID)

Written by the coordinator from the world agent's final report, because the agent's tools refused summary files. H200. "Base" is shared head 387ce372c unless stated. Branch `perf/ovn-world` (local only), 14 commits: 581160519, f4a630def, 62b929bfc, 85ed7b200, 939075f8b, ac29f6d06, adfd6cfbf, 896e9f436, 9579a5b40, 92bb5af04, 37a2ec291, 1b21f1792, c74644ac4, dccfb34b3. One line per experiment in ../../NOTES.md.

## LingBot-World 1.3B (480x832, 93 frames, c=1)
- Default (reference-exact) path: 9.68 → 12.60 fps (+30%); first chunk 1.48 → 1.21 s. Two rounds today.
- The DiT is bit-exact with base. Frames are 69.4 dB vs base; only the VAE decode's accumulation order differs.
- Kept:
  - broadcast the one timestep embedding over the chunk's tokens;
  - channels-last fp32 VAE decode with cuDNN autotuning, compiled and deterministic;
  - camera path computed once per chunk.
- Opt-in `configs/lingbot_world_1_3b_fast.yaml`: 18.6 fps. It uses FA4, compiled pointwise ops, a cuDNN-autotuned encoder and bf16 decode, and drifts 24.6 dB from the default path.
- Dropped:
  - device cache for cu_seqlens;
  - FA4 or cuDNN attention by default (26 dB rollout drift);
  - bf16 decode by default;
  - compiling only the block pointwise ops with norms, GELU and Linear kept eager: 677 → 619 ms per DiT chunk, but not bit-exact (max abs 0.031).

## V-JEPA2-AC (8-frame clip, H=4; 2 rounds)
- c=1: 0.171/0.145 → 0.118 s.
- c=4: 2.9/5.2 → 19.9/16.8 req/s.
- c=8: 8.9/10.2 → 19.5/18.6 req/s.
- Outputs at c=1 are byte-identical to base (6 requests).
- Kept: CUDA-graph capture of the rollout encoder (compile_mode default stays byte-identical; max-autotune moved outputs ~4%), and RoPE tables built once per step.
- The earlier c=4 number (6.5 → 15.9 req/s) was measured before the worker bug fix, on outputs that were wrong on both arms. They were byte-identical between arms at c=1, so the comparison held. Today's numbers are on fixed trees.
- Dropped: compiling the block loop (slower), capture buckets up to 16 (no gain), mixed-iteration batches (c=8 20 → 10 req/s).
- Plateau at ~20 req/s from c=8.

## V-JEPA2 ViT-L
- c=1: 2.19/2.15 → 1.95/1.94 s (in-place normalize; the decode-threads change is in shared). c=4 ~0.79 req/s on both.
- The server is now bound by the API server.
- Lead: clients sending `Accept: application/vnd.mstar.frames` cut a c=1 request from 1.23 to 0.66 s. Client side only.
- The audit's fused-SDPA lead is moot: main already runs cuDNN attention.

## Worker bug fix (merged in shared as 35462fc11 + 7fd1e3589)
- Non-streaming `Loop.accumulated_outputs` (vjepa2 and vjepa2_ac rollouts) were sent from reused pinned host copies. The result was the last iteration's output repeated H times.
- Write-up: `vjepa2_ac_batch_bug.md`.

## Cosmos3 DROID policy (from the pool)
- Kept 85ed7b200: compile the cross-request action batch.
  - Before, concurrent requests batched into an eager step, which was slower than serving them one at a time.
  - Edge, in flight 4: 3.65 → 5.19 chunks/s; 8: 3.55 → 5.43. In flight 1 is unchanged (4.60 vs 4.51; actions bit-identical).
  - Nano, in flight 4: 1.06 → 1.43; 8: 1.14 → 1.48. In flight 1 is 1.24 on both.
  - Batched actions are 0.9% (edge) / 5.7% (nano) rel-L2 from the single-request result. Base's eager batch is 1.0–1.3% / 7.2%.
- Kept 62b929bfc: shared API-server change, a `warmup_requests` key `concurrency: N` (proposal `shared/proposals/world-warmup-concurrency.md`). Test added; the one failure, test_serving_metrics::test_refusals_count..., also fails on base.
  - Edge config f4a630def: first 8-in-flight session 3.12 → 5.33 chunks/s, p90 3.7 → 2.3 s; startup 30 → 42 s.
  - Nano config 581160519 (it had no warmup at all): 0.72 → 1.33 chunks/s, p90 19 → 10 s; startup 36 → 99 s.
- Dropped:
  - CUDA-graph capture of the single-request action_gen step: +2% and not byte-identical (actions 1.05% rel). Hypothesis, not checked: the captured step uses the paged attention backend and the eager one the dense backend.
  - Removing the `.item()` syncs: no change.
- Edge c=1 breakdown (221 ms/chunk): prefill ~8 ms (CPU), VAE encode ~19 ms on the GPU, 4 steps of ~45 ms each (39 ms GPU).
- Remaining leads: overlap the VAE preprocess with the DiT prefill, and resize the observation on the GPU (~4 ms).

## Waypoint
- Plateau, GPU 96% busy: 720p c=1 ~160 fps; 360p c=8 888 → 925/908 fps.
- The branch only fixes the benchmark's request-id parsing (dccfb34b3), so this is neutral.
- Leads: fp8 GEMMs, ring-KV pre-plan.

## Leads for others
- Make binary frame framing the default in the robotics and V-JEPA2 clients.
- Apply the warmup `concurrency` key to other per-batch-size compiled steps.
