# Integrator summary: perf/overnight-shared (overnight 2026-10-20 → 21)

Written by the coordinator from the integrator's final report, because the agent's tools refused summary files. Branch `perf/overnight-shared` (worktree /Users/josephtan/mstar-ovn), base origin/main 55b0215a3, head **a51da9261**, never pushed. All on H200.

## Final shared branch (oldest first)
| sha | change | proposal |
| --- | --- | --- |
| 8ecad036d + 5380b42b1 | MoE fused softmax-topk router, plus its tests | omni-moe-softmax-topk |
| ccc843e40 | Engine `one_row_per_step` piecewise lease | qwen35vl-piecewise-one-row-lease |
| b20071aa4 | Ragged attention opaque custom op + `shard_heads` | video-ragged-opaque-op |
| b9ae42315 + 6e9b686b5 | 8 FFmpeg threads for request video decode | world-load-video-threads |
| ca5324b95 | Conductor least-loaded DP replica | video-dp-least-loaded |
| e270f0b4e | Register opaque ragged callables eagerly (IndexError fix) | coord-ragged-register-eager |
| d1c91505a | Denoise scaffold schedule staging without a stream sync | img-denoise-schedule-nosync |
| aa2cfa4fa | `api_switch_interval_s` config key (0.001 in qwen3_5 configs only) | qwen35-api-switch-interval, resubmitted |
| 35462fc11 + 7fd1e3589 | Held-output host-copy bug fix, plus a drain-test fake fix | world-held-output-host-copy |
| a5fa2ec21 + 387ce372c | Warm every DP replica, plus a serving_metrics stub fix | video-warmup-per-replica |
| a0bba8734 | Fused-MoE swept tiles for Omni's two bf16 shapes | omni-fused-moe-tiles |
| 32db55b6c + a51da9261 | warmup_requests `concurrency` + cosmos3_edge_droid warmups | world-warmup-concurrency |

Not merged:
- img-qwen3-encoder-host-padded: kept on perf/ovn-img (only img's models import it).
- qwen35-api-switch-interval as a global default: rejected, Kokoro c=32 −3.7% in two 4-round runs. Resubmitted and merged as a config opt-in.

Verdicts, with numbers, are appended to each file in shared/proposals/. Every A/B used a fresh server per arm, ABBA order, a correctness check against the base arm, and an MSTAR_PHASE_TIMING profile pass. Two merged proposals each broke a test that stubs worker or API internals by name (test_worker_drain, test_serving_metrics). The box CPU suite caught both, and the integrator fixed them.

## Final combined A/B: head a51da9261 vs main 55b0215a3 (H200, 4 rounds ABBA, fresh servers, host load 6-33 recorded)
| model / metric | delta | notes |
| --- | --- | --- |
| Qwen3.5-0.8B t2t c=64 | **+12.3%** (14.1-16.5k → 16.8-17.8k tok/s) | every head round above every main round |
| Qwen3.5-0.8B c=32 | +11.9% | |
| Qwen3.5-0.8B c=1 | +15.7% | main round 0 ran under load |
| Z-Image B=1 latency p50 | −0.2% | |
| Z-Image c=4 | +1.3% | |
| Qwen3-TTS c=16 | parity | one round per arm halved by a co-tenant and excluded |
| Kokoro c=32 | **−5.5%** | see Open items |
| Kokoro c=1 TTFA | −5% | |

Correctness:
- Qwen3.5: greedy text identical, 8/8 prompts.
- Z-Image: PSNR median 34.6 dB, against a base-vs-base floor of 36.3 / 34.4 dB.
- Qwen3-TTS: WER 0.7% vs 1.6% (it samples, so this is within noise).
- Kokoro: WER identical, 0.94%.
- CPU tests: head and main fail exactly the same 8 timing-sensitive tests.

## Open items
- **Kokoro regression from the held-output fix (35462fc11).** Quiet-host bisect (load 5-9, 4 rounds), head vs head with that fix reverted: Kokoro c=32 −6.9% audio-s/s, +11% TTFA; c=1 and WER unchanged.
  - Hypothesis, not traced: some Kokoro outputs are sent a step after they're produced, and now pay a D2H plus sync on the main thread.
  - The fix stays merged because it corrects data corruption. Follow-up: keep a private host copy only for outputs that are still unsent.
- Omni read-aloud WER in the regression harness is meaningless (Omni doesn't read the prompt verbatim). Use it only as a relative check.

## Noise floor and method
- On shared hosts at load 50-300, Qwen3.5 c=32 swings about ±10% between rounds and Kokoro about ±15%. On a quiet host, about ±4-6%. So: at least 4 ABBA rounds for CPU-bound cells, and deltas under ~4% count as noise.
- Greedy text can differ across server processes of the same build, so the check is "every head output also appears in some main round". It held 8/8 for every merge.

## Files
- Harness: harness/ (regress.py with cells qwen35, qwen35dp2, qwen3tts, kokoro, zimage, omni; cputests.py; cmptests.py; ovn_qwen3_5_0.8b_dp2.yaml).
- Results: box/final/{omni,artts}/ (final run and bisect), ../_box_regress/integrator/, box/{img,regress}/.
