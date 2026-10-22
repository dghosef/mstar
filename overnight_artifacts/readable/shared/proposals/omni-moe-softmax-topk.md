# omni: fused softmax top-k MoE router (shared code, Qwen3-Omni-only user today)

- Branch `perf/ovn-omni`, commit `d48995cb7` ("MoE: softmax top-k routing in one kernel").
- Files: `mstar/model/components/moe.py` (TopKRouter.forward, CUDA path), new `mstar/utils/fused_moe/router.py`
  (`mstar::moe_softmax_topk` custom op, opaque to dynamo).
- Mechanism: `softmax -> torch.topk -> renorm` was 5-6 launches per MoE layer (torch.topk = gatherTopK + bitonicSort,
  ~9 us of ~12 us at decode batch sizes on H200, nsys). One Triton program per token does it in registers.
- Who it affects: only `TopKRouter` users = Qwen3-Omni Thinker (48 layers) and Talker (20 layers). GLM uses its own
  fp8 router; no other model imports these blocks (grep). CPU path unchanged.
- Numbers (H200, qwen3omni_2gpu, 2 rounds alternating, bundled with two model-local Talker changes in arm chg2):
  t2t c=1 215.2 -> 231.2 tok/s (+7.4%), t2t c=16 1170 -> 1228 (+5%); talker-side gains are mixed with the other changes.
- Correctness: not bit-exact. bf16 router logits tie often; torch.topk orders ties arbitrarily (bitonic sort), the
  kernel puts the lower expert first, so equal-weight experts are summed in a different order (sets identical in
  tests). Greedy text at c=1: identical 8/8 run-to-run on each arm, 2/8 identical across arms, first divergence at
  char 82-884; for scale, the unchanged build at c=16 (batching only) diverges from itself on 39/64 prompts at char
  1-1172. Audio WER unchanged (median 0.04-0.08 both arms). Tests: test/modular/test_qwen3_omni_fused_kernels.py.

## Verdict (integrator 23:55): MERGED into perf/overnight-shared as 8ecad036d + 5380b42b1

- Review: it touches only `TopKRouter` (CUDA path) and the new `mstar::moe_softmax_topk` op, and only Qwen3-Omni (Thinker/Talker) imports the MoE blocks, so no other model changes. The kernel is deterministic and breaks ties toward the lower expert id. That is safe under TP: every rank picks the same experts, as with torch.topk. It is graph-capturable (one Triton launch), opaque to dynamo, with a correct fake impl. Output dtype is fp32, the same as the old path. The XPU/CPU path is unchanged. No invariant, S1-S4 or CPU-overhead issues.
- Not bit-exact, by design (tie order, plus fp32 exp vs torch's softmax). I accept that on the proposer's evidence: greedy Thinker text diverges across arms on the order batching alone causes, and audio WER is unchanged.
- Checks (ovn-regress, H200): ruff clean. CPU Core + TP-sim + piecewise-signature vs base 55b0215a3: no new failures. The one extra failure, test_eval_compare::test_stopping_compare_stops_the_job_and_its_servers, also fails on base when run alone (timing flake). Router GPU tests 16/16 pass.
- I moved the two router tests (softmax_topk vs torch, compiled router) from perf/ovn-omni's 3abfc4413 into `test/modular/test_moe_router.py` (commit 5380b42b1), so the shared change carries its own test. **omni: when you rebase, drop those two tests from test_qwen3_omni_fused_kernels.py, or keep them and accept the duplicates.**
- No cross-model suite run: per coordinator, single-importer changes need only that model's check, and the proposer measured Qwen3-Omni (t2t c=1 +7.4%, c=16 +5%).
