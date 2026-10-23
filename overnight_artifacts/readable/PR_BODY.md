> [!CAUTION]
> **DO NOT MERGE. This PR is a tracking board and will never be merged.** It collects every change from an overnight multi-agent performance push (2026-10-20 → 21) so that the changes and results can be seen in one place. Changes that hold up will be split out into separate, focused PRs with full descriptions, tests and review.

<!-- STATUS START -->
<!-- STATUS END -->

## What this is

Eight agents, one per model family, each worked on its own branch and its own H200 box (19 of our Eras H200s). Each change was made after profiling (`MSTAR_PHASE_TIMING` / `benchmark/worker_phases`, plus nsys/torch profiler for GPU-bound models). Each was kept only if an A/B against the base showed a gain and an output check passed (greedy tokens, WER, PSNR, or bit-identity). A ninth agent owned the shared branch: it regression-tested engine/worker/API-server changes on Qwen3.5-0.8B (the CPU-overhead litmus), Qwen3-TTS, Z-Image and Kokoro before merging them.

- Base: `main` at 55b0215a3. All numbers are from an H200, against that base.
- This branch (`perf/overnight-all`) is rebuilt from `perf/overnight-shared` plus every agent branch merged in. It moves forward with a sync commit on each update, so its tree always equals the agents' current branches.
- Caveat: the boxes share physical nodes, so CPU-bound numbers carry about ±10% cross-tenant noise. Small deltas are labelled as noise or re-run, not claimed.
- Out of scope: LTX, pi0.5, Whisper, Nemotron duplex.

## Gains by model (vs main 55b0215a3, H200)

<!-- SUMMARY START -->
<!-- SUMMARY END -->

<!-- LATENCY START -->
<!-- LATENCY END -->

## What was tested

<!-- COVERAGE START -->
<!-- COVERAGE END -->

## Improvements so far (every change)

<!-- TABLE START -->
_See the running log below; this table is regenerated on each update._
<!-- TABLE END -->

## Running log

<!-- LOG START -->
<!-- LOG END -->

🤖 Generated with [Claude Code](https://claude.com/claude-code)
