"""LingBot's Wan2.1 VAE streaming decode: time per 4-latent chunk under several layouts/dtypes,
and PSNR of the uint8 frames against the served config (fp32, NCDHW, 1 latent per call).

  python lb_vae_micro.py --arm /root/work/mstar --out /workspace/results/world/lb_vae
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="/root/work/mstar")
ap.add_argument("--out", required=True)
ap.add_argument("--video", default="/root/work/mstar/test/qwen3-omni/video.webm")
ap.add_argument("--variants", nargs="+", default=["fp32", "fp32_cl", "fp32_lpc0", "fp32_bench", "fp32_cl_bench_lpc0",
                                                  "bf16", "bf16_cl", "bf16_cl_bench_lpc0"])
args = ap.parse_args()
sys.path.insert(0, args.arm)
from diffusers import AutoencoderKLWan  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def load(dtype):
    return AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=dtype).eval().to(dev)


lat_path = out / "latents.pt"
if not lat_path.exists():
    d = VideoDecoder(args.video, num_ffmpeg_threads=8)
    n = 93
    fr = d.get_frames_at(indices=list(range(0, 2 * n, 2))).data  # [T,3,H,W]
    fr = torch.nn.functional.interpolate(fr.float().to(dev) / 127.5 - 1, size=(480, 832), mode="bilinear")
    vae = load(torch.float32)
    with torch.inference_mode():
        z = vae.encode(fr.permute(1, 0, 2, 3)[None]).latent_dist.mode()
    torch.save(z.cpu(), lat_path)
    del vae
z_all = torch.load(lat_path).to(dev)  # [1,16,24,60,104]
print("latents", tuple(z_all.shape))


def run(variant: str):
    dtype = torch.bfloat16 if variant.startswith("bf16") else torch.float32
    vae = load(dtype)
    if "_cl" in variant:
        wan_vae.to_channels_last(vae)
    import re
    m = re.search(r"lpc(\d+)", variant)
    lpc = int(m.group(1)) if m else 1
    step = wan_vae.DecoderStep(vae, compile="compile" in variant, deterministic="det" in variant)
    frames_all, times = [], []
    for rep in range(2):
        cache = wan_vae.new_feat_cache(vae)
        frames = []
        nchunks = (z_all.shape[2] + 3) // 4
        for c in range(nchunks):
            z = z_all[:, :, c * 4:(c + 1) * 4].to(dtype)
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode(), wan_vae.cudnn_benchmark("bench" in variant):
                px = wan_vae.decode_causal(vae, z, cache, step, clip_start=c == 0, latents_per_call=lpc)
                u8 = (px[0] / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)
            torch.cuda.synchronize()
            if c > 0:
                times.append(time.perf_counter() - t)
            frames.append(u8.cpu())
        if rep == 1:
            frames_all = torch.cat(frames, dim=1)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del vae
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = sorted(times[len(times) // 2:])  # second replay only
    return frames_all, {"variant": variant, "ms_per_chunk_med": 1e3 * times[len(times) // 2],
                        "ms_per_chunk_mean": 1e3 * sum(times) / len(times), "peak_gib": peak}


ref = None
results = []
for v in ["fp32", *[x for x in args.variants if x != "fp32"]]:
    fr, res = run(v)
    if ref is None:
        ref = fr
    mse = ((fr.float() - ref.float()) ** 2).mean().item()
    res["psnr_vs_first"] = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    res["bit_exact_vs_first"] = bool(torch.equal(fr, ref))
    results.append(res)
    print(json.dumps(res), flush=True)
(out / "results.json").write_text(json.dumps(results, indent=1))
