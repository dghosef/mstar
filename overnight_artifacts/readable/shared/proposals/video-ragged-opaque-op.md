# video: opaque custom-op RaggedAttentionCallable (+ shard_heads opt-out)

Commit: aa385b022 on perf/ovn-video (engine only; model use in 90a4a56ca).

Mechanism: `RaggedAttentionCallable.__call__` is `@torch.compiler.disable`, so every layer of a
torch.compile'd transformer graph-breaks around attention. Besides the launches, the breaks stop
Inductor fusing pointwise work across them (residual adds, fp32 norms, modulation get materialized).
`RaggedAttentionCallable(resource, label, opaque=True)` instead calls the planned wrapper through a
`mstar::ragged_attention(q, k, v, handle)` custom op (fake impl = empty_like(q)); the compiled region
keeps one graph. Default off: no other model changes behaviour.
Also `RaggedAttentionConfig.shard_heads` (default True): a node whose lockstep ranks split something
other than heads keeps all heads.

Evidence (Wan2.2-TI2V-5B DiT, 1280x704x81, CFG batch 2, H200, standalone torch.profiler):
- cuDNN SDPA in-graph: 1020 ms/forward (attention 478 ms, elementwise ~84 ms)
- FA3 ragged behind graph breaks: 1022 ms (attention 379+15 ms, elementwise ~171 ms)
- FA3 behind the custom op: 958 ms
Served A/B (20 steps, c=1, two servers per arm): dit step 1062 -> 1000 ms. Output PSNR within the
cuDNN-vs-FA2 kernel-swap spread.

Tests: test_ragged_attention.py (new: opaque callable has 0 graph breaks under dynamo.explain and
matches eager; shard_heads=False keeps heads) + ragged/denoise_loop/ltx2_5/resource_runner suites pass.

Could affect (only if they opt in): DenoiseLoopSubmodule users (flux2_klein, z_image: compiled +
captured, so the fusion gain may apply), LTX-2.5 (6 ragged callables in a compiled DiT). Suggest the
integrator / img agent try `opaque=True` in `DenoiseLoopSubmodule.ragged_for` and LTX `attends()`.

## Verdict (integrator 00:05): MERGED into perf/overnight-shared as b20071aa4

- Review: with `opaque=False` (the default) `__call__` dispatches to the same `@torch.compiler.disable` body, so the existing callers (denoise_loop, LTX) behave as before, apart from one extra Python frame that Dynamo traces before the break. `shard_heads` defaults True and is static config, the same on all ranks. The custom op's result depends on plan state the op doesn't take as an input. That is fine here: the plan is fixed for the length of one compiled call, and the op output is fresh, not an alias. Nits: `_RAGGED_CALLABLES` is a process-global list that keeps the callables of earlier resource bindings alive (a few per rebind, bounded in practice); a note in docs/adding_models.rst on when to use opaque=True would help.
- Tests: CPU suite vs base, no new failures. GPU `test_ragged_attention.py` + `test_diffusion_denoise_loop.py` on H200: 79 pass on base, 81 on cand (the 2 new tests pass).
- Regression suite, ovn-regress H200, base 55b0215a3 vs base+aa385b022:
  - Qwen3.5-0.8B t2t, 4 rounds ABBA: c=64 15.0-16.1k vs 15.1-17.0k (+0.8%), c=32 +3.0%, c=1 +1.3%, all noise. Phase profile iter_total 3.15 vs 3.01 ms. Greedy: all cand outputs seen in base (8/8).
  - Z-Image-Turbo 1024^2, B=1 latency 0.859/0.857 vs 0.864/0.854 s (p50 +0.2%); c=4 1.13/1.09 vs 1.07/1.09 img/s. PSNR base-vs-cand median 35.5 dB / min 30.3, against a base-vs-base (cross-process) floor of 33.3 / 28.3. z_image's default config runs SDPA, so this is a no-change check.
- Wan2.2 is the only opt-in user (perf/ovn-video, 90a4a56ca); the proposer measured it (dit step 1062 -> 1000 ms).
