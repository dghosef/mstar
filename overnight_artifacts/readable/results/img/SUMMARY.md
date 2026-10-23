# img agent: BAGEL, FLUX.2-klein, Z-Image-Turbo (final, 14:55 PDT)

Box: ovn-img (2x H200). GPU1 was shared with the integrator from 22:00 to 00:40.
Branch: `perf/ovn-img`, rebased on `perf/overnight-shared` 6e9b686b5. Worktree `/Users/josephtan/mstar-ovn-img`.

Harness (untracked, `benchmark/ovn_img/` in the worktree, mirrored to `/root/work/hx` on the box):

- `run_arm.py` starts a server from a given tree under `benchmark.worker_phases.server` (MSTAR_PHASE_TIMING=20, SHM arena). It runs `benchmark/flux2_klein/bench_images.py`: B=1 latency with the PNGs saved, then closed-loop images/s at c=1/4/8/16 with 2 repeats. With `--runner` it runs `benchmark.runner` for t2t/i2t instead. Then it stops the server and checks the GPU memory came back.
- `compare.py` computes PSNR/SSIM between image sets.
- `nsys_arm.py`, `topk.py`, `gaps.py`, `bursts.py` and `window.py` give the kernel profile and the GPU idle timeline.
- `vit_fp32_check.py` and `vit_attn_micro.py` are offline accuracy and kernel checks.
- `edit_arm.sh` / `edit_client.py` run BAGEL image edits.

Method: fresh servers, arms alternating on one GPU, a shared TORCHINDUCTOR_CACHE_DIR per model (so autotune picks match across arms), at least 2 rounds. Correctness is checked against the base arm at fixed seeds, with a same-code noise floor.

## Kept (commit, oldest first)

| commit | model | change | A/B (H200) | correctness |
| --- | --- | --- | --- | --- |
| 081f762d3 | BAGEL | CFG branch pruning in image_gen (port of perf/bagel-cfg-prune; now `model_kwargs.cfg_prune`, default on). 1 or 2 branches per step instead of 3 | 1024² t2i 16.03/16.03 → 10.10/10.10 s | vs base: 41.4 dB median, 37.6 min, SSIM ≥ 0.989. Same code with a different autotune cache: 40.3/38.0 dB, so this is noise level. Checked by eye |
| a0fb36c42 | klein | Qwen3 text encoder captured per batch size (host-side `text_padded`; rotary tables cached) | c=1 +6%, c=4 +2.7%, c=8/16 +1% | bit-identical (8/8, 2 rounds) |
| 773886861 | Z-Image | config `capture_batch_sizes` [1,2,4,8] → [1..8] | c=4 +15%, c=8 +8%, c=16 +4%, c=1 unchanged, same VRAM | at the model's own restart spread (33.7 dB base vs base) |
| ff4b109d7 | BAGEL | no host syncs in image_gen `prepare_inputs` (BOI/EOI ids and per-shape index tensors built once on device) | prepare_inputs p50 143 → 0.44 ms; 10.34 → 10.21 s | bit-identical |
| 7deb8000b | DiT scaffold (shared; proposal written) | pinned non-blocking staging of the schedule tensors in `denoise_loop` | klein c=4/8/16 +1.2-1.6% (3 rounds), Z-Image +0.3-1.3% | bit-identical |
| bd71ffe43 | Z-Image | Qwen3 text encoder captured, rows padded on the device, output cut in `unpack_packed_outputs` | B=1 −1%, c=1 +1.3%, c≥4 ±0.5% (marginal) | encoder bit-exact eager vs replay (offline); GPU batch-invariance suite passes |
| 3ac1bf456 | BAGEL | ViT (no flash-attn, which is the documented install): unmasked SDPA for a single image; cu_seqlens built once | i2t c=1 TTFT −7%, c=16 +13% tok/s | ViT features vs fp32: old 3.4-4.1e-2 relative error, new the same to 3 digits. Greedy text: 20/64 identical (bf16 drift flips later argmaxes; each arm is itself deterministic) |
| a3a141f74 | BAGEL | parallel CFG (image_gen_cfg) prunes branches too; new `configs/bagel_cfg_parallel_2gpu.yaml` | 2 GPUs: 12.13/12.28 → 6.41/6.48 s; c=4 0.087 → 0.17 img/s | 40.1 dB vs single-GPU base (noise level); deterministic |

## Head vs shared base (6e9b686b5), same box, alternating

- BAGEL t2i (bagel_single_gpu): B=1 16.30/16.36 → 10.16/10.22 s (−38%); c=4 0.061/0.062 → 0.098 img/s (+58%).
- BAGEL t2i (2 GPUs, cfg_parallel_2gpu): 12.1-12.3 → 6.4-6.5 s; c=4 0.087 → 0.17 img/s.
- BAGEL i2t (food101, 64 tokens): c=1 TTFT p50 0.189 → 0.174 s; c=16 448 → 536 tok/s (+20%).
- klein-4B: B=1 0.353/0.364 → 0.343/0.342 s; c=1 2.74/2.73 → 2.82/2.82; c=4 3.28/3.34 → 3.42/3.41; c=8 3.30/3.32 → 3.38/3.36; c=16 3.36/3.35 → 3.42/3.31 img/s.
- Z-Image: running.

## Dropped (with numbers)

- attention_backend flashinfer instead of sdpa (cuDNN) for klein/Z-Image: B=1 the same; klein c=4 −12%, c=16 −9%; Z-Image c=4 −6%, c=16 −4.5%.
- BAGEL per-shape input cache on its own: 10.49/10.33 s vs 10.33-10.41 (no gain until the sentinel sync was also removed; both kept together in ff4b109d7).
- BAGEL image_gen CUDA-graph capture: not pursued. After ff4b109d7 the step stream is seamless (no gap > 0.3 ms between steps) and the GEMMs run at about 720 TFLOP/s.

## Verdicts / plateaus

- BAGEL t2i: GPU-bound (GEMM ~76%, FA3 paged attention 13.5%); seamless after the sync fix.
- BAGEL t2t: decode is GPU-bound (4.6 ms/token; the Inductor Triton GEMVs run at 46-90% of HBM bandwidth). A decode compile-mode A/B is running.
- klein: GPU-bound dit (GEMM 62%, cuDNN SDPA 18%, eager LayerNorm from compile_exact_ops [norms] 8%).
- Z-Image: GPU-bound dit (GEMM 67%, SDPA 18%).

## Not done

FLUX.2-klein-9B is gated (no HF token). bagel_cfg_parallel.yaml needs 3 GPUs; it is replaced by the 2-GPU variant above.

## Afternoon session (14:00-14:55 PDT, ovn-video, 2x H200)

- **FLUX.2-klein-9B** (configs/flux2_klein_9b.yaml, 1024², 4 steps). Baseline (shared base 6e9b686b5): B=1 0.61-0.62 s; c=1 1.59-1.63, c=4 1.70-1.75, c=8 1.71-1.76, c=16 1.71-1.77 img/s. GPU-bound at 90.8% busy: GEMM 68%, cuDNN SDPA 15%, eager LayerNorm (compile_exact_ops [norms]) 6.3%. Head (klein text-encoder capture + sync-free staging; the 9B config already captures every batch size up to 4), median of 3 same-GPU rounds: B=1 0.619 → 0.612 s, c=1 +1.5%, c=4 +1.8%, c=8 +2.3%, c=16 +1.9%. Images are bit-identical to base (8/8) with a shared autotune cache. Cold-autotuned processes differ from each other by 58-63 dB because of the compiled VAE's kernel choice, whatever the code.
- **BAGEL PNG encode** (a6f128346): the 360 ms CPU-only gap per request was PIL's default PNG writer. With the shared parallel writer: B=1 −0.30 to −0.34 s (−3%), c=4 +3%, decoded pixels identical (4 rounds, both GPUs).
- **BAGEL image_gen step capture**: not pursued. Steady-state denoise is 1.0% idle, all in sub-0.1 ms bubbles (warm 3-request nsys), so capture is worth at most ~1%.
- **klein seed noise** (6.4 ms per request on the critical path, ~1% of 9B and ~2% of 4B B=1): torch 2.12's CPU bf16 randn is a float Box-Muller over 24-bit uniforms, then a cast. A vectorized emulation matches all but 66 of 524,288 elements, and those are bf16 rounding-boundary cases, so a bit-exact fast path needs glibc's scalar chain for them and would save at most ~3 ms. Not shipped. Lead: draw the noise ahead of the dit step (ingestion-time prefetch; torch.randn releases the GIL).
