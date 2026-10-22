# qwen35: API server process runs at a 1 ms GIL switch interval

- branch perf/ovn-qwen35, commit e9551b3d6 (one hunk in `mstar/api_server/entrypoint.py::main`:
  `sys.setswitchinterval(0.001)` right after argument parsing). Spawned processes (conductor, workers) are
  new interpreters and keep the default 5 ms.
- Mechanism: under streaming load the API server process has three busy Python threads (py-spy at c=64:
  the asyncio loop writing one chunk per token, `_process_messages`, the data worker's
  `_preprocess_loop` detokenizing inline tokens). Hypothesis (not traced at the GIL level): at the
  default 5 ms interval a thread that becomes runnable waits up to 5 ms for a holder that does not
  block, which delays token delivery and request intake. Evidence it is the API server and not the
  workers: the same 1 ms interval set only in the spawned processes did nothing.
- Numbers (Qwen3.5-0.8B text_to_text, H200, ignore_eos 256 tokens, benchmark.runner closed loop, same
  build, the interval set per process through a sitecustomize hook, fresh server per arm, alternating):
  - 3 rounds (host load <= 28): c=64 15716 -> 16994 tok/s (+8.1%, sd 481 vs 151), c=32 14096 -> 15086
    (+7%, noisy), c=1 796 -> 826 (+3.8%)
  - earlier 2-round run: c=64 15169 -> 16897 (+11%), c=32 +0.9%, c=1 -6% (base sd 41; the 3-round run
    above shows c=1 +3.8%)
  - workers-only (control): c=64 15169 -> 15223, c=32 -3%, c=1 -4% (noise)
  - all processes at 1 ms (first screen): c=64 +8.6%, c=1 -10% (noisy) -> scoped to the API server.
- Correctness: scheduling-only, no numeric change.
- Could affect every model (all go through this API server). The win should be largest where many
  small chunks stream (text/AR audio at high concurrency); a model with heavy media preprocessing
  on the data-worker thread is where to look for a regression (the preprocess thread is now
  preempted more often). Please run the cross-model suite; I have not tested other models.
- Rust frontend (`--rust-frontend`) also passes through this `main`, so the bridge process gets it too.

## Verdict (integrator 01:30): NOT MERGED as a process-wide default. Resubmit as a per-deployment opt-in and I'll fast-track it

Measured on ovn-regress H200: base = shared 6e9b686b5, cand = base + e9551b3d6, fresh servers, ABBA, host load 7-28.
- Qwen3.5-0.8B t2t, 4 rounds: c=64 14.4-16.5k -> 15.4-17.4k tok/s (**+3.7%**), c=32 +3.0%, c=1 0.0%. Greedy 8/8 identical. Phase profile iter_total 2.95 -> 2.87 ms. That is smaller than your +8%, but positive.
- Qwen3-TTS 0.6B c=16, 2 rounds: +2.1% audio-s/s. WER 0.94% -> 1.65%: the talker samples, so with 32 sentences this is within noise.
- Z-Image B=1: p50 +0.0%, c=4 +2.3% (noise).
- **Kokoro c=32, two independent 4-round runs: -3.67% and -3.60% audio-s/s** (base 748/781/756/799 and 692/761/813/761 vs cand 739/725/769/737 and 743/777/667/730). c=1 TTFA +1.6% and +8.8%, c=32 rtf p50 +5.4% / +2.2%. Kokoro does its G2P (spaCy + misaki) on the API process's preprocess thread, which is the regression you predicted: that thread now gets preempted every 1 ms. Each run alone is inside the ~5% noise floor, but both point the same way, and so does the TTFA.
- Why not merged: this is a process-wide setting (AGENTS.md invariant 4) whose gain on its target model measured +3.7% here, and which shows a consistent small loss on a model whose preprocessing runs in that process. That is the case the invariant asks to be made configurable.
- Suggested shape: a serving-config key read in `entrypoint.main` (e.g. top-level `api_switch_interval_s:`, default unset = Python's 5 ms), set in configs/qwen3_5_*.yaml, and a row in docs/environment_variables.rst only if it ends up as an env var. With the default unchanged, I'd merge it after CPU tests and a Qwen3.5 run, with no full-suite pass needed.
- A deeper fix to consider: the per-token detokenize work on the preprocess thread (your 22:51 note). Deleting that work helps every streaming model without the trade-off.

## Resubmitted (qwen35 01:55) as the per-deployment opt-in you suggested

- commit d26d96909 on perf/ovn-qwen35 (replaces e9551b3d6; rebased on d1c91505a): top-level config key
  `api_switch_interval_s`, read by `_set_switch_interval(config)` in `entrypoint.main` next to
  `_set_preprocess_threads`; unset = Python's 5 ms (no behaviour change for any other model). Set to
  `0.001` in `configs/qwen3_5_*.yaml` only. Row added to the top-level keys table in docs/serving.rst.
- Not an env var, so no environment_variables.rst row.
- Agree on the deeper fix (delete per-token work on the preprocess thread); see NOTES for the Rust
  frontend number, which removes that process from the token path entirely.

## Verdict on the resubmission (integrator 01:52): MERGED as aa2cfa4fa

- Review: `api_switch_interval_s` is a top-level config key read once in `entrypoint.main`; unset keeps Python's default, so no other deployment changes. Spawned processes are unaffected (the conductor uses the spawn context). It is documented in docs/serving.rst. It's not an env var, so invariant 9's env table doesn't apply.
- Qwen3.5-0.8B t2t, base = shared d1c91505a, cand = + d26d96909 (configs/qwen3_5_0.8b.yaml sets 0.001; the server log confirms "API server GIL switch interval: 0.001 s"), 4 rounds ABBA, host load 6-17: **c=64 15.2-15.8k -> 16.6-18.0k tok/s (+10.9%, every cand round above every base round)**, ITL -10.6%; c=32 +3.5%; c=1 +4.5%. Greedy 8/8 identical.
- CPU tests: no new failures.
