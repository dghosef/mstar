#!/usr/bin/env python3
"""Code-predictor GEMM shapes: eager F.linear (cuBLAS) vs inductor max-autotune, graph-timed."""
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

SHAPES = {"qkv": (4096, 1024), "o_proj": (1024, 2048), "gate_up": (6144, 1024), "down": (1024, 3072),
          "lm_head": (2048, 1024)}
torch.manual_seed(0)
lin = torch.compile(F.linear, dynamic=False, mode="max-autotune-no-cudagraphs")
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    for m in (1, 8, 32):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        lin(x, w)
        te = do_bench_cudagraph(lambda: F.linear(x, w), rep=50) * 1e3
        tc = do_bench_cudagraph(lambda: lin(x, w), rep=50) * 1e3
        print(f"{name:8s} M={m:3d} N={n} K={k}: cublas {te:6.2f}us  inductor {tc:6.2f}us  "
              f"roofline {n*k*2/4.8e12*1e6:5.2f}us", flush=True)
