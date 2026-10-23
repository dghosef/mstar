# Overnight perf push: raw artifacts (2026-10-20 → 21)

Everything that survived the run, for anyone who wants the details behind the tables in the PR description. Not code. Nothing here is imported or run by mstar.

## Layout
- `readable/`: browse on GitHub.
  - `readable/*.md` at the top: PLAYBOOK.md (the rules the agents worked under, with timestamped updates), NOTES.md (one line per experiment, kept and dropped, from every agent), LOG.md (the coordinator's running log), PR_MAIN.md (the PR description), CLAIMS.md, and the older tables (TABLE.md, SUMMARY_TABLE.md, COVERAGE.md).
  - `readable/shared/proposals/*.md`: every shared-code proposal, with the integrator's verdict and numbers appended.
  - `readable/results/<agent>/SUMMARY.md`: each agent's final write-up (qwen35, qwen35vl, omni, artts, tts, img, video, world, integrator).
  - `readable/results/<agent>/harness/`: the benchmark and A/B drivers each agent wrote.
  - `readable/results/_box_<box>/`: text copied off each GPU box before it was terminated.
  - Patches for dropped ideas (`*_dropped.patch`, `*_untested.patch`), configs, CSVs and small result JSON.
- `archive/overnight_results_full.tar.xz` (~20 MB; extract with `tar -xJf`): every surviving text artifact, deduplicated. That's ~10.5k files: all benchmark result JSON, phase-timing and server logs, per-request outputs, and profile summaries.

## What is not here
- Audio, images, video and profiler traces (nsys, torch) lived on the GPU boxes' volumes, which were deleted after the run.
- Per-request transcript files from mirrored directories were dropped as duplicates.
- Credentials: the HF token, ssh config and Eras endpoint credentials were never stored here. Both layers were scanned before commit.
