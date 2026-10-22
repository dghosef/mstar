# omni: swept fused-MoE tiles for Qwen3-Omni's two bf16 shapes (shared code, Omni-only user today)

- Branch `perf/ovn-omni`, commit `eb2c45a12` ("fused MoE: swept tiles for Qwen3-Omni's two shapes at small batch").
- Files: `mstar/utils/fused_moe/kernels.py` (`_TUNED` table + `get_tuned_configs`), `mstar/utils/fused_moe/runner.py`
  (bf16 `fused_experts` picks gate/up and down tiles separately, same BLOCK_SIZE_M for both).
- Mechanism: the bf16 path used one default tile (16x32x64) for both GEMMs at every small M. Table rows only for
  (E=128, N=1536, K=2048) up to 8 tokens (Thinker) and (E=128, N=768, K=1024) up to 4 (Talker); any other shape,
  TP-sharded N, or larger M falls through to the default unchanged. fp8 path untouched.
- Microbench (H200, CUDA-graph timed, random routing): Thinker M=1 gate+down 23.0 -> 17.6 us/layer, M=2 38.6 -> 34.8,
  M=4 81.6 -> 75.0, M=8 125.4 -> 115.8; Talker M=4 16.1 -> 14.0.
- Served A/B (ab3, 2 rounds alternating, bundled with two model-local Talker changes): t2t c=1 230.5 -> 239.9 tok/s
  (+4%; t2t is Thinker-only, so this is the tiles), t2t c=16 flat.
- Who it affects: only callers of bf16 `fused_experts` with those exact shapes = Qwen3-Omni (grep: no other importer of
  the MoE blocks). Correctness: not bit-exact (different K blocking), greedy-text divergence comparable to batching noise.

## Verdict (integrator 12:45): MERGED into perf/overnight-shared as a0bba8734

- Review: the table hits only exact (E, N, K) = (128, 1536, 2048) for M<=8 and (128, 768, 1024) for M<=4, which are Qwen3-Omni's Thinker and Talker. Any other shape, TP-sharded N or larger M falls through to `get_default_config` unchanged, and the fp8 path isn't touched. The gate/up and down GEMMs share BLOCK_SIZE_M, so the one token alignment stays valid. Static tables, no per-step cost beyond one dict lookup. No other importer of the bf16 `fused_experts` shapes (grep).
- Tests (ovn-omni H200, the merged head): test_fused_moe_decode + test_fused_moe_prefill + test_qwen3_omni_fused_moe + test_moe_router + test_startup_warmup: 168 passed. CPU Core + TP-sim + piecewise-signature on the head vs main 55b0215a3: identical failure sets (8 timing-flaky eval_compare/serving_metrics tests on both).
- Not bit-exact (K blocking). I accept that on the proposer's greedy-divergence-vs-batching argument. Served gain is the proposer's: Omni t2t c=1 +4%. No cross-model suite run, since the change is single-importer (coordinator policy).
