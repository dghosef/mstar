"""Wan VAE whole-clip decode timing: latents_per_call x cudnn.benchmark, via the shared decoder.

CUDA_VISIBLE_DEVICES=0 python vae_bench.py --repo Wan-AI/Wan2.2-TI2V-5B-Diffusers --t 21 --h 44 --w 80
(Cosmos3: --repo <snapshot>/vae subfolder handled by --subfolder vae)
"""
import argparse
import time

import torch
from diffusers import AutoencoderKLWan

from mstar.model.components import wan_vae

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
ap.add_argument("--t", type=int, default=21)
ap.add_argument("--h", type=int, default=44)
ap.add_argument("--w", type=int, default=80)
ap.add_argument("--per-call", default="1,2,4,0")
ap.add_argument("--bench", default="0,1")
ap.add_argument("--stock", action="store_true", help="also time compiled stock vae.decode")
args = ap.parse_args()

vae = AutoencoderKLWan.from_pretrained(args.repo, subfolder="vae").eval().to("cuda", torch.bfloat16)
z = torch.randn(1, vae.config.z_dim, args.t, args.h, args.w, device="cuda", dtype=torch.bfloat16)


def timeit(fn, n=3):
    fn()
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000, out


with torch.inference_mode():
    if args.stock:
        dec = wan_vae.compile_decode(vae.decode)
        ms, ref = timeit(lambda: dec(z).sample)
        print(f"stock compiled vae.decode: {ms:.0f} ms", flush=True)
    with torch.inference_mode(False):
        wan_vae.to_channels_last(vae)
    step = wan_vae.DecoderStep(vae, compile=True, deterministic=True)
    base = None
    for bench in (int(b) for b in args.bench.split(",")):
        for per_call in (int(p) for p in args.per_call.split(",")):
            with wan_vae.cudnn_benchmark(bool(bench)):
                ms, out = timeit(lambda: wan_vae.decode_causal(
                    vae, z, wan_vae.new_feat_cache(vae), step, clip_start=True, latents_per_call=per_call))
            if base is None:
                base = out
            err = (out.float() - base.float()).abs().max().item()
            print(f"fast decode per_call={per_call} cudnn_bench={bench}: {ms:.0f} ms  maxdiff_vs_first={err:.3g}"
                  f"  peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            torch.cuda.reset_peak_memory_stats()
