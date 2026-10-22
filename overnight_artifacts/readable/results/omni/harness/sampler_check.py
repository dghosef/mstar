#!/usr/bin/env python3
"""small_vocab_sample vs the FlashInfer graphable path: distribution and time."""
import torch
from triton.testing import do_bench_cudagraph

import flashinfer
from mstar.engine.resources.sampler.small_vocab import small_vocab_sample
from mstar.engine.resources.sampler.utils import fused_temperature_softmax

torch.manual_seed(0)
dev = "cuda"


def fi(logits, t, k, p, seed, off):
    probs = fused_temperature_softmax(logits, t, include_greedy=True)
    k = torch.where(k > 0, k, logits.shape[1])
    return flashinfer.sampling.top_k_top_p_sampling_from_probs(probs, k, p, deterministic=True,
                                                               seed=seed, offset=off).to(torch.int64)


V = 2048
for (temp, topk, topp) in [(1.0, 50, 0.8), (0.9, 50, 1.0), (1.0, 0, 0.9), (0.7, 20, 0.95)]:
    N = 200000
    row = torch.randn(V, device=dev) * 2.5
    logits = row.expand(N, V).contiguous().to(torch.bfloat16)
    t = torch.full((N,), temp, device=dev)
    k = torch.full((N,), topk, device=dev, dtype=torch.int32)
    p = torch.full((N,), topp, device=dev)
    seed = torch.arange(N, device=dev, dtype=torch.int64) * 7 + 1
    off = torch.arange(N, device=dev, dtype=torch.int64)
    a = small_vocab_sample(logits, t, k, p, seed, off)
    # FlashInfer folds the row into philox, so one call gives N independent draws
    b = fi(logits, t, k, p, seed, off)
    ha = torch.bincount(a, minlength=V).float() / N
    hb = torch.bincount(b, minlength=V).float() / N
    # the exact target: softmax, top-k, renorm, top-p (minimal set), renorm
    pr = torch.softmax(row.to(torch.bfloat16).float() / temp, -1)
    if 0 < topk < V:
        th = pr.topk(topk).values[-1]
        pr = torch.where(pr >= th, pr, 0)
        pr = pr / pr.sum()
    if topp < 1:
        sv, si = pr.sort(descending=True)
        keep = (sv.cumsum(0) - sv) < topp
        m = torch.zeros_like(pr)
        m[si[keep]] = 1
        pr = pr * m
    pr = pr / pr.sum()
    tv = lambda x, y: 0.5 * (x - y).abs().sum().item()
    print(f"T={temp} k={topk} p={topp}: TV(new,exact)={tv(ha, pr):.4f} TV(flashinfer,exact)={tv(hb, pr):.4f} "
          f"TV(new,fi)={tv(ha, hb):.4f} support new={int((ha > 0).sum())} fi={int((hb > 0).sum())} exact={int((pr > 0).sum())}")

# greedy rows are argmax
lg = torch.randn(64, V, device=dev).to(torch.bfloat16)
z = torch.zeros(64, device=dev)
g = small_vocab_sample(lg, z, torch.full((64,), 50, device=dev, dtype=torch.int32), torch.full((64,), 0.8, device=dev),
                       torch.zeros(64, device=dev, dtype=torch.int64), torch.zeros(64, device=dev, dtype=torch.int64))
print("greedy == argmax:", torch.equal(g, lg.float().argmax(-1)))

for bs in (1, 8, 32):
    lg = torch.randn(bs, V, device=dev).to(torch.bfloat16)
    t = torch.ones(bs, device=dev)
    k = torch.full((bs,), 50, device=dev, dtype=torch.int32)
    p = torch.full((bs,), 0.8, device=dev)
    s = torch.zeros(bs, device=dev, dtype=torch.int64)
    o = torch.zeros(bs, device=dev, dtype=torch.int64)
    ta = do_bench_cudagraph(lambda: small_vocab_sample(lg, t, k, p, s, o), rep=50) * 1e3
    tb = do_bench_cudagraph(lambda: fi(lg, t, k, p, s, o), rep=50) * 1e3
    print(f"bs={bs}: new {ta:.2f}us  flashinfer path {tb:.2f}us")
