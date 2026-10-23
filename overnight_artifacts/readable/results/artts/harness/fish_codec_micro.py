#!/usr/bin/env python3
"""Eager and graph time of the Fish S2-Pro codec decode per (batch, window).

    PYTHONPATH=/root/work/base python fish_codec_micro.py [--tf32]
"""
import argparse
import contextlib
import time

import torch

from mstar.model.fish_s2_pro.submodules import full_fp32
from mstar.model.registry import get_model_class, model_init_kwargs

ap = argparse.ArgumentParser()
ap.add_argument("--tf32", action="store_true", help="codec_exact_fp32=False")
ap.add_argument("--bs", default="1,2,4,8,16")
ap.add_argument("--cudnn-bench", action="store_true")
ap.add_argument("--widths", default="4,12,28,60,124,252,508,1058")
args = ap.parse_args()

kw = model_init_kwargs("fish_s2_pro")
kw["codec_exact_fp32"] = not args.tf32
model = get_model_class("fish_s2_pro")(**kw)
sub = model.get_submodule("Codec", device="cuda:0")
dev = "cuda:0"
torch.backends.cudnn.benchmark = args.cudnn_bench
nq = model.config.num_codebooks


def timed(fn, n=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


with torch.inference_mode():
    for width in [int(w) for w in args.widths.split(",")]:
        for bs in [int(b) for b in args.bs.split(",")]:
            codes = torch.randint(0, 1000, (bs, nq, width), device=dev)
            start = torch.zeros(bs, dtype=torch.long, device=dev)
            conv_start = torch.full((bs,), max(0, width - sub.conv_frames(width)), dtype=torch.long, device=dev)
            guard = full_fp32 if not args.tf32 else contextlib.nullcontext
            total = timed(lambda: sub._decode(codes, start, conv_start))

            def lat():
                with guard():
                    positions = start.view(-1, 1) + torch.arange(width, device=dev)
                    return sub.decoder.latents(codes, positions)
            t_lat = timed(lat)
            latents = lat()
            cf = sub.conv_frames(width)
            idx = (conv_start.view(-1, 1) + torch.arange(cf, device=dev)).clamp(max=width - 1)
            g = latents.gather(2, idx.unsqueeze(1).expand(-1, latents.shape[1], -1))

            def conv():
                with guard():
                    return sub.decoder.run_conv_stack(g)
            t_conv = timed(conv)
            # graph replay of the whole decode (fixed shapes)
            g2 = torch.cuda.CUDAGraph()
            for _ in range(2):
                sub._decode(codes, start, conv_start)
            torch.cuda.synchronize()
            with torch.cuda.graph(g2):
                sub._decode(codes, start, conv_start)
            t_graph = timed(g2.replay)
            del g2
            print(f"width={width:5d} conv={cf:3d} bs={bs:3d} eager={total:7.2f} ms  graph={t_graph:7.2f} ms  "
                  f"latents={t_lat:6.2f} conv={t_conv:6.2f}", flush=True)
