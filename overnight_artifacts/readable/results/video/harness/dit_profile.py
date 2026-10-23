"""Standalone Wan2.2 DiT step profile: SDPA vs FlashInfer ragged self-attention.

CUDA_VISIBLE_DEVICES=1 python dit_profile.py --mode sdpa|fa3|auto [--profile] [--batch 2]
Run from the tree to test (cwd), with that tree on PYTHONPATH.
"""
import argparse
import time

import torch

from mstar.model.wan22.components.dit import Wan22AttentionFns
from mstar.model.wan22.config import Wan22Config
from mstar.model.wan22.weight_loader import build_wan22_dit

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="sdpa")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--frames", type=int, default=21)
ap.add_argument("--h", type=int, default=44)
ap.add_argument("--w", type=int, default=80)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--nocompile", action="store_true")
ap.add_argument("--opaque", action="store_true")
ap.add_argument("--autotune", action="store_true")
ap.add_argument("--perframe", action="store_true")
args = ap.parse_args()

cfg = Wan22Config()
dev = torch.device("cuda")
dit = build_wan22_dit(cfg, "Wan-AI/Wan2.2-TI2V-5B-Diffusers", device=dev)
if not args.nocompile:
    dit.set_fuse_ffn_gelu(True)
    dit.forward = torch.compile(dit.forward, fullgraph=False, dynamic=False,
                                mode="max-autotune-no-cudagraphs" if args.autotune else None)
B = args.batch
S = args.frames * (args.h // 2) * (args.w // 2)
x = torch.randn(B, 48, args.frames, args.h, args.w, device=dev, dtype=torch.bfloat16)
ts = torch.full((B, args.frames if args.perframe else S), 500.0, device=dev)
txt = torch.randn(B, 512, 4096, device=dev, dtype=torch.bfloat16)

fns = None
if args.mode != "sdpa":
    import flashinfer
    ws = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=args.mode)
    cu = torch.arange(0, B + 1, dtype=torch.int32) * S
    w.plan(cu, cu, 24, 24, 128, causal=False, q_data_type=torch.bfloat16, sm_scale=128 ** -0.5)
    print("flashinfer backend:", getattr(w, "_backend", "?"))

    if args.opaque:
        @torch.library.custom_op("ovn::ragged", mutates_args=())
        def _op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return w.run(q, k, v)

        @_op.register_fake
        def _(q, k, v):
            return torch.empty_like(q)

        def self_attn(q, k, v):
            return _op(q, k, v)
    else:
        @torch.compiler.disable
        def self_attn(q, k, v):
            return w.run(q, k, v)

    fns = Wan22AttentionFns(self_attn=self_attn, cross_attn=None)

with torch.inference_mode():
    for _ in range(3):
        dit(x, ts, txt, attn_fns=fns)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        dit(x, ts, txt, attn_fns=fns)
    torch.cuda.synchronize()
    print(f"mode={args.mode} B={B} S={S}: {(time.perf_counter() - t0) / args.iters * 1000:.1f} ms/forward")
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            dit(x, ts, txt, attn_fns=fns)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=90))
