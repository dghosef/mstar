#!/usr/bin/env python3
"""Tile table for cp_linear: per (N, K) and M, the best skinny tile if it beats cuBLAS by >5%."""
import itertools

import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph

from mstar.model.qwen3_omni.components.cp_kernels import _skinny_linear_kernel


def run(x, w, y, bn, bk, warps, stages):
    m, k = x.shape
    n = w.shape[0]
    _skinny_linear_kernel[(triton.cdiv(n, bn),)](x, w, y, m, n, k, x.stride(0), w.stride(0), y.stride(0),
                                                 BM=16 if m <= 16 else 32, BN=bn, BK=bk,
                                                 num_warps=warps, num_stages=stages)


SHAPES = {"qkv": (4096, 1024), "o_proj": (1024, 2048), "gate_up": (6144, 1024), "down": (1024, 3072),
          "lm_head": (2048, 1024)}
torch.manual_seed(0)
table = {}
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    for m in (1, 2, 4, 8, 16, 32):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x, w)
        tc = do_bench_cudagraph(lambda: F.linear(x, w), rep=50) * 1e3
        best = None
        for bn, bk, warps, stages in itertools.product([8, 16, 32, 64], [64, 128, 256], [2, 4, 8], [2, 3, 4, 6]):
            if k % bk or (bn < 16 and m > 16):
                continue
            try:
                run(x, w, y, bn, bk, warps, stages)
                if not torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=2e-2):
                    continue
                t = do_bench_cudagraph(lambda: run(x, w, y, bn, bk, warps, stages), rep=30) * 1e3
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, (bn, bk, warps, stages))
        win = best[0] < 0.95 * tc
        print(f"{name:8s} M={m:3d}: cublas {tc:5.2f}us skinny {best[0]:5.2f}us {best[1]} {'USE' if win else ''}",
              flush=True)
        if win:
            table.setdefault((n, k), []).append((m, best[1]))
print("TABLE", table)
