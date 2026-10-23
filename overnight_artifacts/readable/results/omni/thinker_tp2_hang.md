# Qwen3-Omni thinker-TP2 on 2 GPUs: startup hang is pre-existing on main 55b0215a3

**Verdict: main hangs the same way. It is not caused by perf/ovn-omni.** It is a startup-ordering problem: a TP
barrier runs per capture spec, and one TP rank also warms non-TP nodes first. Whether it fires depends on
compile-cache warmth, not on the code under test.

## Config (`configs/qwen3omni_thinker_tp2_2gpu.yaml`, guest-only, not committed)
qwen3omni_thinker_tp2.yaml squeezed onto 2 GPUs:
```yaml
resources: {thinker_kv: {max_num_pages: 1024}}
node_groups:
  - {node_names: [Thinker], ranks: [0, 1], tp_size: 2}
  - {node_names: [audio_encoder, vision_encoder], ranks: [0]}
  - {node_names: [Talker], ranks: [1]}
  - {node_names: [Code2Wav], ranks: [1]}
```
`mstar serve qwen3_omni --config configs/qwen3omni_thinker_tp2_2gpu.yaml --gpus 0,1` on ovn-artts (2x H200).

## Runs (ovn-artts, results under /workspace/results/omni/{tp2a,tp2hang}/)
| run | tree | inductor cache | outcome |
| --- | --- | --- | --- |
| tp2a/head_r0 (18:41 UTC) | branch HEAD | cold for this box | NCCL watchdog: rank 0 ALLREDUCE (barrier) SeqNum=2 timed out at 600 s; worker_0 SIGABRT |
| tp2hang/base_dbg (20:29) | main 55b0215a3 | warm (base's Talker/Code2Wav graphs were already compiled by an earlier, aborted base run on this box) | passed the barrier: rank 1 finished Code2Wav+Talker warmup in ~2 min and reached the Thinker specs (stopped by hand afterwards) |
| tp2hang/base_cold (20:42) | main 55b0215a3 | cold (`TORCHINDUCTOR_CACHE_DIR` = empty dir) | **hang**: `[Rank 0]: Ranks 1 failed to pass monitoredBarrier in 600000 ms` (TORCH_DISTRIBUTED_DEBUG=DETAIL turns the barrier into gloo monitoredBarrier) at 20:52:54; worker_0 exited, DeadWorkerError |

All three runs had NCCL_DEBUG=INFO and TORCH_DISTRIBUTED_DEBUG=DETAIL. On the main cold run, NCCL init completed on both ranks
(`ncclCommInitRankConfig ... nRanks 2`). Nothing failed at the transport level.

## Where each rank is stuck (py-spy dump, main cold run, 20:44 UTC)
- **Rank 0** (worker_0, cuda:0; nodes `['Thinker', 'audio_encoder', 'vision_encoder']`): parked in the per-spec TP
  barrier for the Thinker's first capture spec, there since ~20:43:
  `barrier (communication.py:119) <- size_captures (accelerator_graph_runner.py:820) <- _plan_captures (engine.py:766) <- warmup (engine.py:683) <- warmup_all <- Worker.run (worker.py:4174)`
- **Rank 1** (worker_1, cuda:1; nodes `['Code2Wav', 'Talker', 'Thinker']`): still warming the **Talker**, a TP=1
  node, inside Inductor autotuning:
  `... Talker forward (talker.py:110/161/226) <- _forward_decode_like (submodules.py:1424) <- forward_batched (submodules.py:1530) <- run_forward/_warmed (accelerator_graph_runner.py:1126/1140) <- size_captures (accelerator_graph_runner.py:824) <- _plan_captures (engine.py:766)`
  Its Code2Wav capture logged at 20:43:12 and talker_prefill at 20:52:26. It never reached the Thinker's barrier before rank 0's 600 s ran out.

On the head run (tp2a), rank 0's NCCL flight recorder shows the identical stack (`barrier <- size_captures:820 <- _plan_captures:766 <- warmup:683`). Rank 1 logged Talker captures from 18:49 to 18:57, past the 18:52 timeout.

## Mechanism (code is the same in main and the branch)
`Engine.warmup` -> `_plan_captures` iterates `self._submodules`, the nodes *this worker* hosts, in their own order.
For each capture spec, `AcceleratorGraphRunner.size_captures` calls `self._comm_group.tp_group.barrier()` before warming
that spec ("every rank sizes these specs, in config order"). Rank 0's first node is the Thinker, so it enters a Thinker
TP barrier at once. Rank 1 first warms and sizes Code2Wav and the Talker, which are not in the TP group and so have no
barrier. A cold Inductor max-autotune compile of the Talker takes ~10-15 min. The NCCL (or gloo-monitored) barrier
timeout is 600 s, so rank 0 gives up. With a warm compile cache rank 1 gets there in ~2 min and startup succeeds.
This is an instance of invariant 11 (lockstep is also about *when*): the TP ranks execute the same collectives in the same order. But one rank interleaves minutes of non-collective work between them, and nothing bounds it below the collective timeout.

The branch's per-rank changes (MoE router/tiles, Talker row outputs) are not involved. The hang is at warmup, before any
request, and on main too. The branch only changes how long the Talker takes to compile (it is cold either way on a new box).

## Fix directions (not implemented)
- Warm and size the nodes of TP groups first on every rank (sort `_submodules` so lockstep nodes lead in `_plan_captures` and in the capture loop), so non-TP compile time comes after the last TP barrier.
- Or give warmup barriers a startup-sized timeout: a CPU (gloo) barrier with an explicit timeout, or a longer `timeout=` on the TP process group during warmup.
- Supported layouts today keep TP ranks free of other nodes (qwen3omni_thinker_tp2.yaml puts the Talker on rank 2), and those are unaffected.
