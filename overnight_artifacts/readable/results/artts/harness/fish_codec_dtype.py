#!/usr/bin/env python3
"""Fish codec decode in exact fp32 vs TF32 vs bf16: SNR against exact, and time.

    PYTHONPATH=<tree> CUDA_VISIBLE_DEVICES=1 python fish_codec_dtype.py
"""
import contextlib
import copy
import time

import torch

from mstar.model.fish_s2_pro.submodules import full_fp32
from mstar.model.registry import get_model_class, model_init_kwargs

model = get_model_class("fish_s2_pro")(**model_init_kwargs("fish_s2_pro"))
sub = model.get_submodule("Codec", device="cuda:0")
dev = "cuda:0"
nq = model.config.num_codebooks
dec32 = sub.decoder
dec16 = copy.deepcopy(dec32).to(torch.bfloat16)


def decode(dec, codes, start, conv_start, cf, mode):
    guard = full_fp32() if mode == "exact" else contextlib.nullcontext()
    with guard:
        if mode == "bf16":
            positions = start.view(-1, 1) + torch.arange(codes.shape[-1], device=dev)
            lat = dec.latents(codes, positions)
            idx = (conv_start.view(-1, 1) + torch.arange(cf, device=dev)).clamp(max=codes.shape[-1] - 1)
            lat = lat.gather(2, idx.unsqueeze(1).expand(-1, lat.shape[1], -1))
            return dec.run_conv_stack(lat).float()
        return dec.decode_window(codes, start, conv_start, cf).float()


def timed(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


torch.manual_seed(0)
with torch.inference_mode():
    for width, bs in [(60, 1), (60, 8), (124, 16)]:
        codes = torch.randint(0, 1000, (bs, nq, width), device=dev)
        start = torch.zeros(bs, dtype=torch.long, device=dev)
        cf = sub.conv_frames(width)
        conv_start = torch.full((bs,), width - cf, dtype=torch.long, device=dev)
        ref = decode(dec32, codes, start, conv_start, cf, "exact")
        out = {}
        for mode, dec in (("tf32", dec32), ("bf16", dec16)):
            y = decode(dec, codes, start, conv_start, cf, mode)
            snr = 10 * torch.log10(ref.pow(2).sum() / (y - ref).pow(2).sum().clamp_min(1e-12)).item()
            ms = timed(lambda: decode(dec, codes, start, conv_start, cf, mode))
            out[mode] = (snr, ms)
        ms_exact = timed(lambda: decode(dec32, codes, start, conv_start, cf, "exact"))
        print(f"width={width} bs={bs} exact={ms_exact:.1f} ms | tf32 {out['tf32'][1]:.1f} ms snr {out['tf32'][0]:.1f} dB"
              f" | bf16 {out['bf16'][1]:.1f} ms snr {out['bf16'][0]:.1f} dB", flush=True)
