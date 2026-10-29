# Qwen3-TTS push notes (2026-10-29)
Profiles, findings and the SLO bench harness from the Qwen3-TTS performance push on top of perf/ovn2-final (4e3de8eae).
- PROFILE.md / FINDINGS.md: bottleneck breakdown and per-agent findings (code branches: qwen3tts/{depth,codec,prefill,host,close-gap,base} on this fork)
- READY.md: commits ready to integrate; BASE.md: the combined base (perf stack + host-overhead stack #849–#861)
- profile/: worker_phases tables (lead: closed-loop Seed-TTS c=1/16/32/64; host: per-thread CPU + phases)
- harness/: open-loop Poisson SLO client (TTFA + stall), cell runner, Whisper WER / speaker-sim eval

## Workloads

- `data/passages.jsonl`: the passages workload, 6 English passages (7, 7, 29, 48, 79, 100 words), cycled; ~17.6 s of audio per request on average.
- Short sentences: Seed-TTS `en/meta.lst` (seed-tts-eval), 1,088 sentences, shuffled.
