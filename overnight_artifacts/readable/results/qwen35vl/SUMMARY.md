# qwen35vl summary (final, 13:58 PDT)

Agent qwen35vl. Overnight on ovn-qwen35vl (1x H200); resumed 12:00-13:45 on ovn-world/ovn-video as a guest.
Branch perf/ovn-qwen35vl (worktree /Users/josephtan/mstar-ovn-qwen35vl), rebased on perf/overnight-shared
387ce372c. Numbers: back-to-back arms, fresh server per arm, alternating, >=2 rounds unless noted; H200.
Harness: results/qwen35vl/harness (own box) and gharness (guest boxes): ovn.py server driver, qwen_arm.py /
asr_arm.py / garm.py arms, qcheck.py greedy correctness + per-image-size TTFT, table.py.

## Commits (on top of 387ce372c)
| sha | change |
| --- | --- |
| 173dae9af | Qwen3.5: request images stay uint8 into the processor; raw image no longer shipped to workers |
| (ccc843e40, merged to shared) | Engine: `one_row_per_step` lease for piecewise regions that pack a step into one row |
| 671f48c15 | Qwen3.5: vision encoder batches <=8 requests/step into ONE captured ViT replay; embed/merger compiled in-node (engine whole-forward compile recompiled per batch size, ~0.6 s each on live requests); image grids ride step metadata (no device syncs in prepare_inputs); host-built grid tensors, pinned non-blocking copies |
| bea7ddc28 | Qwen3.5: ship byte patches (processor without rescale/normalize), normalize on device; bitwise-same pixels |
| c978e4656 | Omnilingual ASR: compile the conv feature extractor (dynamic, warmed at load) |
| f68bd555e | Omnilingual ASR: packed varlen encoder via the ragged attention resource + piecewise capture per frame bucket |
| 4b4f55240 | Omnilingual ASR: positional conv as per-group unfold+bmm for wide groups (1b/3b/7b); cuDNN picks slow legacy kernels there |
| (video branch) f9bd589d0, 070e636ce | the Qwen3.5 changes re-done on origin/main's video path (see below) |
| 10a8b64f0 | Qwen3.5: drop the MSTAR_QWEN35_VISION_BATCH A/B knob (constant 8); refresh configs' warmup comment |

## Qwen3.5 image_to_text (food101 ~200 img tok, 128 out tok)
| size | metric | base | head |
| --- | --- | --- | --- |
| 0.8B | i2t c32 tok/s | 5480 / 6027 / 6034 / 5906 | 7596 / 7830 / 8370 / 8118 / 8572 (+34-40%) |
| 0.8B | i2t c32 TTFT p50 | 168-191 ms | 73-121 ms |
| 0.8B | t2t c32 tok/s | 9.2k-11.7k | 11.3k-12.1k (no regression) |
| 0.8B | c1 TTFT 1536x1152 / 2048^2 / 2560^2 / 4032x3024 | 60-62 / 184-190 / 266-398 / - | 41-43 / 85-95 / 146-153 / 433-456 ms |
| 0.8B | mixed t2t+i2t c16+c16 tok/s | 3894+3815 / 4061+3845 | 4900-5349 + 4643-5118 (+25-30%) |
| 2B | i2t c32 tok/s, TTFT p50 | 4924 / 4950, 229-237 ms | 6731 / 6853 (+38%), 113-122 ms |
| 4B | i2t c32 tok/s, TTFT p50 | 3318 / 3325, 316-329 ms | 4406 (+33%), 179 ms (pre byte-patch head) |
| 9B | i2t c32 tok/s, TTFT p50 (1 round) | 2554, 424 ms | 3180 (+25%), 271 ms |
t2t unchanged at every size (2B 10.3-10.7k both; 4B 6.2-6.4k vs 6.1k; 9B 4746 vs 4809).
Correctness (CORRECTED): the uint8-image and byte-patch changes are bitwise-identical in model input (pixels
compared across trees; CPU tests). The vision-batching commit (in-node dynamic compile of the ViT embed/merger)
is NOT byte-identical at c=1: 16-17/26 greedy outputs match base, vs 26/26 base-vs-base (on main: 9/14 vs
14/14). Its embed/merger match an fp32 reference as closely as the engine-compiled ones (rel 2.3e-3 / 3.4e-3),
so this is bf16 reordering, not a bug, but a task-accuracy check (e.g. MMMU/Video-MME) is owed before a PR.
Batched c=8 outputs differ from c=1 at the same rate on base and head.
Profile: base ran 32 single-request ViT steps per wave of 32 image requests (5-8 ms host each, prepare_inputs
syncs), ~60% of wall; head ~4 per wave; image steps now GPU-bound. PLATEAU (see NOTES).
video_to_text (main 63e950dc3, #427): branch perf/ovn-qwen35vl-video = origin/main (abdfd5802) + 7c72a70e0
(one_row_per_step lease, from shared) + f9bd589d0 (vision batching incl. video frame groups, segment ceiling
per step, host grids, uint8 images) + 070e636ce (image and video byte patches, device normalize). Conflicts
were re-done by hand on #427's code, not merged: main's new can_lease stays and now caps the step's total
segments. Qwen3.5 CPU tests pass (90; video tests compare after the device normalize; one new can_lease case).
0.8B, synthetic 8 s 360p clips, 2 rounds, main -> branch:
| video c1 req/s (TTFT p50) | 5.8-5.9 (116-120 ms) -> 6.4-7.0 (91-92 ms) |
| video c8 req/s (TTFT p50) | 8.9 (675-687 ms) -> 14.7-14.8 (440 ms), +66% |
| video c32 req/s (TTFT p50) | 11.4-11.7 (2.7 s) -> 15.8-15.9 (1.9 s), +36% |
| image c32 tok/s (TTFT p50) | 5.8-6.0k (177-187 ms) -> 7.4-7.5k (73-96 ms) |
| text c32 tok/s | 10.2k -> 10.7k (noise) |
c1 TTFT 8 s 720p 362 -> 275 ms, 16 s 360p 209 -> 167 ms. Rust ext rebuilt for the 17 new commits
(/root/guest-qwen35vl/rustext on ovn-video, via PYTHONPATH).

## Omnilingual ASR CTC (LibriSpeech test-clean 500, RTFx = audio s / wall s; WER gate)
300m, 2 rounds vs base: c1 380-385 -> 519-527; c16 1649-1731 -> 2179-2202; c64 2006-2026 -> 2468-2472; WER 5.28-5.31% both.
1b vs base: c1 178 -> ~385 (2.2x); c16 804 -> ~1550; c64 1047 -> ~1940; WER 3.40% -> 3.40%.
3b vs base: c1 ~113 -> 305-333 (2.8x); c16 470 -> ~890; c64 582 -> ~1000; WER 2.37% -> 2.37% (c1 2.01 -> 1.91%).
7b vs base: c1 ~97 -> 198-228 (2.2x); c16 286 -> 494-535 (+80%); c64 305 -> 520-534 (+73%); WER 2.63% -> 2.61%.
Positional-conv commit alone (clean sequential, 2 rounds, vs the previous head): 7b c16 +22%, 3b c16 +33%,
c64 neutral (-2%), c1 2.2-2.8x on 1b/3b/7b.
Mechanisms (each A/B'd separately): conv frontend fp32 channel-LayerNorm on transposed copies was ~45% of a
batched 300m step (compiled: c64 +23%); packed varlen + captured encoder removed padding/masked SDPA and
launches (300m c1 +37%, 1b-7b c64 +76-89%); cuDNN legacy kernels for the wide-group 128-tap positional conv
were ~85% of 7b c=1 GPU time (35 ms of ~41 ms per clip) -> unfold+bmm (c1 2.2-2.8x on 1b-7b).
c64 at 300m is API-process bound (~330 req/s; two client processes give the same total).

## Dropped (numbers in NOTES)
Qwen: prefill_vision cap 8 (+5% tok/s, +15% TTFT); capture buckets to 8192/16384 tok (no gain / worse);
bf16 pixel_values over SHM (no change). ASR: whole-encoder compile (-5..-10%); DP length-aware sub-batching
(no served gain); compile=True on captured encoder, opaque or not (no gain); float attn mask (-2.5% enc only);
frontend length/row padding + cuDNN warmup (no gain); frontend convs as unfold+GEMM (+8% c16, WER +0.05);
tap-accumulated fp32 GEMM for the positional conv (slower: 35 vs 21 ms at 16x1000); api_switch_interval_s on
the ASR config (no change).

## Open leads
- ASR 7b/3b c64: unfold+bmm copies K=128x the input (8 GB at 16x1000 frames, ~20 ms) where cuDNN's lucky
  shapes take 4.6 ms; a no-copy tap-accumulated GEMM in fp32 was slower (35 ms). A proper implicit-GEMM
  (Triton) grouped conv would win both.
- ASR c64 API process: multipart parse + upload written to disk and re-read, single preprocess thread
  (the api_switch_interval_s opt-in did not help here).
- Qwen: prefill_vision rows in the mixed decode+prefill walk so image prefills stop stalling decode.
