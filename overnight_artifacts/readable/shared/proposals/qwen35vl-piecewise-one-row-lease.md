# Proposal: piecewise regions can lease a whole step as one row (`one_row_per_step`)

- Branch `perf/ovn-qwen35vl`, commit `7e4b0ba61` (engine part only: accelerator_graph_config.py, accelerator_graph_runner.py, engine.py; +11 lines).
- Consumer: `0d3a81904` (Qwen3.5 vision encoder batching) on the same branch.

## Mechanism
`Engine._maybe_lease_piecewise_regions` leases a piecewise region's slot at `bs = len(inputs)`. A packed region
that replays all of a step's requests as segments of ONE row (Qwen3.5's ViT block loop: every image of every
request is its own ragged-attention segment, captured at bs=1) then finds no bucket for bs>1 and the step runs
eagerly. New opt-in `PiecewiseAcceleratorGraphConfig.one_row_per_step` (default False) makes the lease use bs=1.
With the default nothing changes for any other model (the only code path touched is the lease call, behind the flag).

## Numbers (H200, Qwen3.5-0.8B, image_to_text c=32, food101, 128 out tokens; base = 55b0215a3)
text tok/s: base 5480 / 6027, head 7596 / 7830 (two rounds, ABAB, fresh servers); TTFT p50 186 -> 74..105 ms.
text_to_text c=32 unchanged within noise (9.2k-12.1k across all arms both sides).
Profile: per burst of 32 image requests base ran 32 single-request vision steps (~5-8 ms host each); head runs 4.

## Affects
Only configs that set the flag (Qwen3.5 vision_encoder). Other piecewise users (vjepa2, qwen3_tts, cosmos3) unchanged.
Regression check suggested: any model with `lease_before_step=True` (vjepa2, qwen3_tts) smoke run.

## Verdict (integrator 00:05): MERGED into perf/overnight-shared as ccc843e40

- Review: an opt-in flag on PiecewiseAcceleratorGraphConfig (default False), and one conditional in `Engine._maybe_lease_piecewise_regions`. With the default it does nothing for any model, and the lease decision comes from static config, so it is identical on every rank (invariant 11). Nits, none blocking: a region that sets `one_row_per_step` without `lease_before_step` silently ignores it, which could be asserted at config time; there is no unit test for the flag; and docs/adding_models.rst could mention it next to lease_before_step.
- CPU tests (base 55b0215a3 vs cand): no new failures (the one extra, test_eval_compare::test_stopping_compare..., fails on base run alone).
- Regression suite, ovn-regress H200, 4 rounds ABBA, fresh server per arm, base 55b0215a3 vs base+7e4b0ba61:
  - Qwen3.5-0.8B t2t (256 out, ignore_eos, greedy). Base round 0 overlapped CPU tests on the box, so judge by rounds 1-3. c=64 base 15.6-16.3k vs cand 14.9-16.3k tok/s (mean -1.3%, noise); c=32 13.1-13.9k vs 13.6-14.3k (+4.8%, noise); c=1 +2.2%. Phase profile at bs~61: iter_total 2.978 vs 2.979 ms, event_sync 0.10 both.
  - Greedy text: every cand output also appears in some base round (8/8). Base rounds disagree among themselves 4/8 across processes (kernel selection).
  - Qwen3-TTS 0.6B c=16 (lease_before_step path): 101.4/102.4 vs 102.6/103.1 audio-s/s (+0.95%), WER 0.94% vs 0.47%, 0 failures.
