# coord: register opaque ragged callables eagerly (bug fix for merged b20071aa4)

Branch perf/ovn-coord, commit e270f0b4e (on shared head ca5324b95).

**Bug:** a node that builds its opaque RaggedAttentionCallable lazily on first forward (e.g.
qwen35vl's Omnilingual ASR packed encoder) builds it inside the compiled region. Dynamo traces
`_register_ragged`: the list append is not replayed, but the handle it read (len-1 of the traced
list) is baked into the graph, so every call raises IndexError. Reproduced on CPU in one cold
process (no cache involved): 4/4 calls IndexError with 0 callables registered.

**Fix:** `@torch.compiler.disable` on `_register_ragged`, so registration always runs eagerly
(one graph break at construction only; calls through the custom op are unchanged).

**Evidence:** new GPU test test_opaque_callable_built_inside_a_compiled_region fails without the
fix, passes with it; test/modular/test_ragged_attention.py 56/56 on H200 (ovn-regress GPU1); the CPU
repro (~/runs/overnight/coord/ragged_repro.py) passes cold and with a warm FX cache. ruff clean.

**Affects:** only opaque=True callables (Wan2.2 constructs them eagerly, so it doesn't change);
unblocks qwen35vl using opaque=True under compile.

## Verdict (integrator 01:25): MERGED into perf/overnight-shared (fast-forward to e270f0b4e)

- Review: the root cause is right. A traced `_register_ragged` bakes `len(list)-1` into the graph but doesn't replay the append. `@torch.compiler.disable` makes registration eager, at the cost of one graph break when the callable is built and none per call. An eagerly built callable (Wan2.2) is unaffected. A lazy builder under `fullgraph=True` would now get a loud compile error instead of a silent wrong handle, which is the better failure.
- Tests on ovn-regress H200 (GPU3): test_ragged_attention.py + test_ragged_attention_cpu.py + test_diffusion_denoise_loop.py, 96 passed. ruff clean.
- No suite run: the change only affects opaque=True callables, and no default config on the shared head builds one.
- Suite note: compile+capture coverage for ragged users is the new GPU test plus these files, which I now run for every ragged/engine-attention proposal (no default served model uses opaque ragged yet).
