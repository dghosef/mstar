# img: DiT scaffold schedule staging without a stream sync

- branch perf/ovn-img, commit 7deb8000b
- File: `mstar/model/components/diffusion/denoise_loop.py` (`_step_scalar_sources`). The sigma/timestep tensors now go through `pin_memory().to(device, non_blocking=True)` instead of a pageable `.to(device)`.
- Mechanism: the pageable H2D copy synchronizes the stream, so on every new request's first denoise step the GPU thread waited for all queued denoise work (klein at bs~6.6: gpu_thread.prepare_inputs p95 531 ms, mean 56 ms). Values are unchanged.
- Numbers (H200, alternating arms, shared inductor cache): klein c=4 +1.6%, c=8 +1.2%, c=16 +1.2% over 3 clean rounds, c=1 unchanged; Z-Image c=4 +1.3%, c=8 +0.9%, c=16 +0.3% over 2 rounds. klein served images are bit-identical.
- Affects every DenoiseLoopSubmodule user: flux2_klein, z_image, ltx2_5 (LTX is out of scope tonight; same code path, so no correctness risk, but untested here).
- CPU tests: `pytest test/modular -k "denoise or diffusion or klein or z_image or ltx"` gave 213 passed. test_waypoint_dit segfaults on CPU on the base tree too (pre-existing, unrelated).

## Verdict (integrator 01:45): MERGED into perf/overnight-shared as d1c91505a

- Review: this is the fix AGENTS.md invariant 7 asks for (a pageable per-request H2D in the step path is a sync). The pinned temporary is safe with non_blocking=True: PyTorch's caching host allocator records the copy's stream event before it reuses the block. Non-accelerator devices keep the plain `.to()`. Values are unchanged.
- CPU tests vs base 55b0215a3: no new failures (7 -> 7).
- Z-Image-Turbo cell on ovn-regress H200, base = shared 6e9b686b5, 3 rounds ABBA: c=4 1.085/1.089/1.079 -> 1.096/1.100/1.085 img/s (+0.8%); B=1 p50 0.860/0.854/0.855 -> 0.851/0.858/0.852 s (-0.3%); gpu_thread.prepare_inputs mean 0.316 -> 0.224 ms. PSNR base-vs-cand median 34.7 dB, against a cross-process base floor of 33.3 (z_image isn't repeatable across restarts; see img 23:55).
- Users: flux2_klein and z_image (img, measured by the proposer and here), LTX-2.5 (out of scope; same code path, values unchanged).
