# Overnight paper-quality benchmark brief (2026-10-21 → 22)

Goal: publication-quality comparison of M* against vLLM and SGLang (and their omni / diffusion variants where the model needs them), for the models improved in the overnight perf PR (dghosef/mstar#13, branch perf/overnight-all).

## Arms
- **mstar-pr:** branch perf/overnight-all. Today 642b44668. A code review is fixing it now, and the coordinator will message the final sha. Measure this arm LAST, on the final sha.
- **mstar-main:** upstream main abdfd5802 (the PR's base), to show the PR's own delta.
- **vLLM, SGLang:** latest stable releases, pinned. vllm-omni / sglang-omni / SGLang diffusion / diffusers where the base engine can't serve the model.

## Rules (the mstar-bench skill has the details, so load it first)
- **Matched configs.** Same dtype (bf16), max model length, max batch / concurrency limit, CUDA graphs on, prefix caching set the same way on all arms (off, unless a workload needs it), same chunked-prefill budget where configurable. Tune each competitor honestly with its own recommended flags, and record every flag.
- **Warmup waves before every measured run.** At least 3 rounds per point across fresh server restarts, with engine order rotated per round. Report mean and min–max. Every point needs failed == 0, or it is reported as failed.
- **Full concurrency sweep:** c = 1, 2, 4, 8, 16, 32, 64 (128 where an engine still scales).
  - Throughput metrics: output tok/s (LLM), audio-s/s or RTF (TTS/ASR), img/s or videos/min (diffusion).
  - Latency metrics: TTFT / TTFA p50 and p99, plus TPOT p50 and p99 for LLMs.
- **Datasets.** Use standard ones, not the ones tuned on overnight, to reduce overfitting:
  - text: ShareGPT-style real prompts, plus fixed random in/out 1024/256;
  - vision: a public VQA / image set;
  - video: real clips of several lengths;
  - TTS / ASR: LibriSpeech / LibriTTS text.
- **Correctness gate per model before timing:** outputs compared against the model's reference (and across arms) per the mstar-correctness rules. A speedup without a passing gate isn't reported as a win.
- **Profile pass for mstar wherever it trails a competitor:** MSTAR_PHASE_TIMING=100 rendered with benchmark/worker_phases, with a CPU- vs GPU-bound verdict.

## Models, in priority order (cover as many as the night allows, depth before breadth)
1. Qwen3.5 text: 0.8B, 9B (1 GPU), 9B TP2. 27B TP4 if a 4-GPU window is free.
2. Qwen3.5 image→text and video→text (0.8B, 9B).
3. Qwen3-Omni-30B-A3B (vllm-omni, sglang-omni): text→text, text→speech, audio→text.
4. TTS: Qwen3-TTS, Fish S2-Pro, Orpheus, against whichever engines serve them.
5. Diffusion: Wan2.2, Z-Image-Turbo, FLUX.2-klein, against vllm-omni / SGLang diffusion / diffusers.

## Box
- `ssh -F ~/runs/bench1021/ssh_config bench` (4×H200 now).
- Tree at /root/work/mstar (perf/overnight-all, see COMMIT.txt); setup log /root/work/setup.log; HF_HOME=/root/work/hf, with a read-only HF token at /root/work/hf/token (never print or copy it).
- Results go to /workspace/results/bench/. rsync TEXT results to ~/runs/bench1021/results/ after every phase: md/json/csv/log, no media or traces, because the Mac disk is small.
- The coordinator may move you to an 8-GPU node once one frees up. Until the coordinator confirms the box (or 60 min pass with no word), do setup, engine installs, downloads, smoke tests and correctness gates only, no timed rounds.

## Added (user): publication rigor, competitors on their best foot
- Latest stable vLLM / SGLang, pinned versions and commits.
- Per model, start from each project's own recommended config, then a short tuning sweep of its key knobs, keeping the best:
  - vLLM: max-num-seqs, max-num-batched-tokens, gpu-mem-util, async scheduling, cudagraph sizes, attention backend, compile level;
  - SGLang: max-running-requests, chunked-prefill-size, cuda-graph-max-bs, mem-fraction-static, attention backend, schedule policy, torch-compile.
- Verify competitor health: no eager fallback, graphs captured, no missing-kernel warnings.
- Cross-check one point per model with `vllm bench serve` / `sglang.bench_serving`.
- Same API, streaming mode, tokenizer settings and ignore_eos for fixed lengths on all arms.
- The tuning sweeps and final flags go in REPORT.md.
- Fewer models done right beats more models done loosely.
- One model at a time on the box. During timed runs only that server runs.

## Box update
The final box is **node8**: a dedicated 8×H200 p5en node, `ssh -F ~/runs/bench1021/ssh_config node8`. All timed runs happen there. The 4-GPU "bench" slice is temporary and gets terminated once the agent says it's done with it.

## Second node (added 03:50 UTC Oct 22): diffusion lane
- A second dedicated 8×H200 node, `node2` in ~/runs/bench1021/ssh_config, runs the **diffusion models only** (Wan2.2, Z-Image-Turbo, FLUX.2-klein), with their own agent. The node8 agent drops diffusion from its queue and keeps text → vision → Omni → TTS.
- mstar-pr sha for every lane: **d248477ec** (perf/overnight-all head after the review fixes). mstar-main: abdfd5802.
- Same rules: one model at a time per node, competitors tuned first, warmup + ≥3 rounds across restarts with rotated order, failed==0, correctness gate (PSNR/SSIM vs the reference pipeline, cross-process floor), profile pass where mstar trails.
- Diffusion competitors: diffusers (reference, with torch.compile + its recommended attention backend as the tuned arm), vllm-omni, SGLang diffusion — whichever serve the model. Record versions and flags.
- Metrics: B=1 latency (mean, p50, p99) at the standard resolution/steps for each model, and throughput (img/s or videos/min) at c = 1, 2, 4, 8 (16 for images), failed==0.
- Results: node2:/workspace/results/bench/<model>/, text-only rsync to ~/runs/bench1021/results/diffusion/. Write ~/runs/bench1021/REPORT_DIFFUSION.md; the coordinator merges it into REPORT.md.

## Lane split (05:00 UTC, user)
- node8: Qwen3.5 text (0.8B, 9B, 9B TP2, 27B TP4) → Qwen3.5 image/video → Qwen3-Omni.
- node2: diffusion (Z-Image-Turbo, FLUX.2-klein, Wan2.2), then TTS (Qwen3-TTS 1.7B, Fish S2-Pro, Orpheus, Qwen3-TTS 0.6B).
