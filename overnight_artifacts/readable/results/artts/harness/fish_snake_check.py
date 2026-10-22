#!/usr/bin/env python3
"""Fused vs eager Snake1d in the Fish codec: output difference and conv-stack time.

    PYTHONPATH=<tree with fused Snake1d> python fish_snake_check.py
"""
import time

import torch

import mstar.model.fish_s2_pro.components.dac_layers as L
from mstar.model.registry import get_model_class, model_init_kwargs

model = get_model_class("fish_s2_pro")(**model_init_kwargs("fish_s2_pro"))
sub = model.get_submodule("Codec", device="cuda:0")
dev = "cuda:0"
tri = L.triton


def timed(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


torch.manual_seed(0)
with torch.inference_mode():
    for width, bs in [(60, 1), (60, 4), (60, 16), (12, 8)]:
        codes = torch.randint(0, 1000, (bs, model.config.num_codebooks, width), device=dev)
        start = torch.zeros(bs, dtype=torch.long, device=dev)
        conv_start = torch.full((bs,), max(0, width - sub.conv_frames(width)), dtype=torch.long, device=dev)
        L.triton = None
        ref = sub._decode(codes, start, conv_start)
        t_ref = timed(lambda: sub._decode(codes, start, conv_start))
        L.triton = tri
        out = sub._decode(codes, start, conv_start)
        t_new = timed(lambda: sub._decode(codes, start, conv_start))
        diff = (out.int() - ref.int()).abs()
        print(f"width={width} bs={bs} eager={t_ref:.2f} ms fused={t_new:.2f} ms  "
              f"int16 max|diff|={diff.max().item()} frac_nonzero={(diff > 0).float().mean().item():.2e}", flush=True)
