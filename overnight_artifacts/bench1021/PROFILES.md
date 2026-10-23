## Profiles (mstar-pr, `MSTAR_PHASE_TIMING=100`)

### What the profiles say, across models

The profile pass ran with `MSTAR_PHASE_TIMING=100`, on mstar-pr only, at the cells where M* trails its strongest competitor most (plus c=1 when it trails there). The findings below are general, not model-specific. The biggest costs aren't inside the per-step GPU work: the worker often waits for requests, and much of the rest goes to tiny batches and per-request Python.

Each item is marked **measured** (from the phase tables), **traced** (read in code) or **hypothesis**. Every estimate is unconfirmed until A/B tested.

1. **The API server can't feed the worker fast enough.** Measured; the cause is traced.
   - **Evidence:** on Qwen3.5-0.8B the worker is idle 28–32% of the time on long prompts. On ShareGPT c=128 it only ever holds 47 of the 128 requests, and they arrive in bursts. TTFT is 590 ms against SGLang's 40 ms, while throughput is close. The bigger models are 96–99.9% busy.
   - **Traced cause:** one API-server thread handles every output token one at a time, detokenizing each separately, and takes at most one new request per pass.
   - **Hypothesis:** requests also hold their engine slots until a late round-trip finishes, which could cause the bursts.
   - **Fix:** detokenize per frame for all rows at once; give admission its own time budget or process; free engine slots as soon as a request's last step finishes; make the Rust frontend the default once its c=1 regression is fixed.
   - **Estimate:** up to +37–47% on small text and image/video models. It would also help Omni, Orpheus, Higgs and Omnilingual ASR.
2. **Tiny side-walk steps take most of the steps on TTS models.** Measured.
   - **Evidence:** side walks (the codec or vocoder stage, run as its own step between decode steps) are 1–3-row steps that cost about as much as a full 60-row decode step. They are 66% of steps on Qwen3-TTS 1.7B, 49% on Orpheus and 46% on Higgs.
   - **Traced cause:** the scheduler is round-robin and gives up its turn after one step.
   - **Fix:** a per-walk "minimum rows or maximum wait" declared by the model, or a pick by rows × priority with an age limit. First-chunk rows keep priority, to protect time to first audio.
   - **Estimate:** up to 1.65–1.7× in theory, +30–50% realistically. This targets the Qwen3-TTS gap to SGLang-Omni and Orpheus at high concurrency.
3. **Keep per-row loop state on the GPU.** Measured.
   - **Evidence:** when a decode loop carries embeddings or codes per row, rather than just tokens, the worker spends ~2.5 ms per step routing them through Python and preparing inputs. That hits Qwen3-TTS, Fish and the Omni talker.
   - **Fix:** a declared engine resource that gathers and scatters those rows on the GPU.
   - **Estimate:** +20–40% on Qwen3-TTS at high concurrency. Large change.
4. **Per-row Python cost in the decode step.** Measured.
   - **Evidence:** on 0.8B a decode step costs 1.28 ms + 31 µs per row, almost none of it GPU time. The per-row cost is in planning, the sampler's commit and postprocessing, each rebuilding Python lists and small tensors every step.
   - **Fix:** columnar per-step state, and extend last step's plan instead of rebuilding it.
   - **Estimate:** up to +60% in decode-heavy stretches on small models, visible only once item 1 is fixed. First check whether the 31 µs is work or GIL wait.
5. **Image models lose time outside the worker.** Measured.
   - **Evidence:** ~80–85 ms per request at c=1 on Z-Image and FLUX.2-klein, and ~54 ms on OmniVoice. That is roughly the whole c=1 gap to SGLang, so those gaps are mostly not kernel time.
   - **Hypothesis:** PNG/WAV encoding, base64, conductor round-trips and the API server reading output tensors.
   - **Next step:** a per-request timeline before fixing anything.
6. **Cap packed prefill tokens at the largest captured graph size.** Measured and traced.
   - **Evidence:** on 0.8B a 16-row × 1024-token prefill runs as ~16k tokens, past the largest captured graph (8k), so it runs eagerly. 9B video prefill also runs past its largest captured size.
   - **Fix:** cap tokens per step at the largest captured bucket, as well as rows.
   - **Estimate:** +5–10%, and lower tail latency. Small to medium change.
7. **Fold new prefills into the running decode step.** Traced.
   - **Evidence:** the existing `speculation_mixes_walks` option is off everywhere, so every prefill step stalls the whole decode batch.
   - **Next step:** an A/B with it on, on Omni ASR first, whose time per output token is 2× SGLang-Omni's.
8. **Copy only what the stop check needs back to the CPU.** Measured and traced.
   - **Evidence:** the worker copies every GPU output to the CPU each step: 56 ms of a 240 ms step on Omnilingual 7B.
   - **Estimate:** up to +25–30% there. Small change.
9. **Profiler gaps.** These block confirming the rest.
   - No time is recorded while the worker sits idle, so item 1 doesn't show in the phase tables.
   - Nothing separates GIL waits from work.
   - There are no spans in the API server or conductor.

**What this data can't show:** GPU kernel efficiency (the 9B/27B long-prompt gap, and the Z-Image/FLUX time inside the GPU), anything on Qwen3-Omni ASR (only its text→text path was profiled), and GIL waits versus work.

**Three measurements would settle most of it:** py-spy on the API server, conductor and worker together (Omni ASR, and 0.8B at c=128); nsys on 9B long prompts, Z-Image single-image and Qwen3-TTS c=1; per-thread CPU time per phase on 0.8B and Qwen3-TTS.

Items 1, 2, 5 and 8 look like the best value for the effort.

### Phase tables per model

Each link is the worker phase table for the profiled cell (named by workload and concurrency; the text models' tables cover several cells). `event_sync` is the only real GPU wait.

| model | phase tables |
|---|---|
| bagel | [edit_c4](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/bagel/profile/phases_edit_c4.txt) |
| bagel_2gpu | [t2i_c2](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/bagel_2gpu/profile/phases_t2i_c2.txt) |
| chatterbox | [c8](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/chatterbox/profile/phases_c8.txt) |
| cosmos3_edge | [t2v_c4](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/cosmos3_edge/profile/phases_t2v_c4.txt) |
| cosmos3_edge_droid | [policy_c4](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/cosmos3_edge_droid/profile/phases_policy_c4.txt) |
| cosmos3_nano | [t2v_c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/cosmos3_nano/profile/phases_t2v_c1.txt) · [t2v_c4](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/cosmos3_nano/profile/phases_t2v_c4.txt) |
| fish_s2pro | [c8](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/fish_s2pro/profile/phases_c8.txt) |
| higgs_audio | [ls500_c32](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/higgs_audio/profile/phases_ls500_c32.txt) |
| klein4b | [c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/klein4b/profile/phases_c1.txt) |
| klein9b | [c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/klein9b/profile/phases_c1.txt) |
| kokoro | [c4](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/kokoro/profile/phases_c4.txt) |
| lingbot_world | [rollout_c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/lingbot_world/profile/phases_rollout_c1.txt) |
| omniasr_1b | [ls500_c64](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/omniasr_1b/profile/phases_ls500_c64.txt) |
| omniasr_300m | [ls500_c64](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/omniasr_300m/profile/phases_ls500_c64.txt) |
| omniasr_3b | [ls500_c64](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/omniasr_3b/profile/phases_ls500_c64.txt) |
| omniasr_7b | [ls500_c64](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/omniasr_7b/profile/phases_ls500_c64.txt) |
| omnivoice | [c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/omnivoice/profile/phases_c1.txt) · [c64](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/omnivoice/profile/phases_c64.txt) |
| orpheus | [c64](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/orpheus/profile/phases_c64.txt) |
| q35_0.8b | [mstar-pr](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/q35_0.8b/phases_mstar-pr.txt) · [profile_verdict](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/q35_0.8b/profile_verdict.txt) |
| q35_0.8b_vl | [mstar-pr-rf](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/q35_0.8b_vl/phases_mstar-pr-rf.txt) |
| q35_27b_tp4 | [mstar-pr](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/q35_27b_tp4/phases_mstar-pr.txt) |
| q35_9b | [mstar-pr](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/q35_9b/phases_mstar-pr.txt) |
| q35_9b_tp2 | [mstar-pr](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/q35_9b_tp2/phases_mstar-pr.txt) |
| q35_9b_vl | [mstar-pr](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/q35_9b_vl/phases_mstar-pr.txt) |
| qwen3_omni | [mstar-pr](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/qwen3_omni/phases_mstar-pr.txt) |
| qwen3tts_0p6b | [c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/qwen3tts_0p6b/profile/phases_c1.txt) · [c32](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/qwen3tts_0p6b/profile/phases_c32.txt) |
| qwen3tts_1p7b | [c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/qwen3tts_1p7b/profile/phases_c1.txt) · [c64](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/qwen3tts_1p7b/profile/phases_c64.txt) |
| vjepa2 | [clip_c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/vjepa2/profile/phases_clip_c1.txt) |
| wan22 | [c2](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/wan22/profile/phases_c2.txt) |
| waypoint | [rollout_c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/waypoint/profile/phases_rollout_c1.txt) |
| waypoint_360p | [rollout_c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/waypoint_360p/profile/phases_rollout_c1.txt) |
| zimage | [c1](https://github.com/dghosef/mstar/blob/perf/overnight-all/overnight_artifacts/bench1021/results/diffusion/zimage/profile/phases_c1.txt) |
