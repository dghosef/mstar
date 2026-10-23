#!/usr/bin/env python3
"""Skinny GEMM (M <= 32) for the code predictor's projections vs cuBLAS, graph-timed.

y = x @ W.T, x (M, K) bf16, W (N, K) bf16. One program per BN output columns (and an
optional K split reduced with fp32 atomics into a zeroed buffer + a cast).
"""
import itertools
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph


@triton.jit
def _skinny_kernel(X, W, Y, M, N, K, sxm, swn, sym,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * sxm + (k0 + rk)[None, :], mask=(rm[:, None] < M), other=0.0)
        w = tl.load(W + rn[None, :] * swn + (k0 + rk)[:, None], mask=(rn[None, :] < N), other=0.0)
        acc = tl.dot(x, w, acc)
    tl.store(Y + rm[:, None] * sym + rn[None, :], acc.to(Y.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def skinny(x, w, BN, BK, warps, stages):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    BM = 16 if M <= 16 else 32
    _skinny_kernel[(triton.cdiv(N, BN),)](x, w, y, M, N, K, x.stride(0), w.stride(0), y.stride(0),
                                          BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return y


SHAPES = {"qkv": (4096, 1024), "o_proj": (1024, 2048), "gate_up": (6144, 1024), "down": (1024, 3072),
          "lm_head": (2048, 1024)}
torch.manual_seed(0)
for name, (n, k) in SHAPES.items():
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    for m in [int(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "8", "32"])]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ref = F.linear(x, w)
        tc = do_bench_cudagraph(lambda: F.linear(x, w), rep=50) * 1e3
        best = None
        for bn, bk, warps, stages in itertools.product([8, 16, 32, 64], [64, 128, 256], [2, 4, 8], [2, 3, 4, 6]):
            if bn < 16 and m > 16:
                continue
            try:
                y = skinny(x, w, bn, bk, warps, stages)
                if not torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=2e-2):
                    continue
                t = do_bench_cudagraph(lambda: skinny(x, w, bn, bk, warps, stages), rep=30) * 1e3
            except Exception:
                continue
            if best is None or t < best[0]:
                best = (t, (bn, bk, warps, stages))
        print(f"{name:8s} M={m:3d}: cublas {tc:5.2f}us  skinny {best[0]:5.2f}us {best[1]}  "
              f"roofline {n*k*2/4.8e12*1e6:4.2f}us", flush=True)
