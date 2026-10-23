#!/usr/bin/env python3
"""FlashInfer paged decode, tensor-core vs CUDA-core, for Qwen3-Omni's Thinker (32q/4kv) and
Talker (16q/2kv) heads, page 128, KV rotating over 64 caches so L2 doesn't serve it."""
import torch
import flashinfer
from triton.testing import do_bench, do_bench_cudagraph

PAGE, D = 128, 128
ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
for name, hq, hkv in (("thinker", 32, 4), ("talker", 16, 2)):
    for bs in (1, 8, 32):
        for kvlen in (300, 1000):
            pages_per = (kvlen + PAGE - 1) // PAGE
            n_layers = 64
            caches = [torch.randn(bs * pages_per, 2, PAGE, hkv, D, device="cuda", dtype=torch.bfloat16)
                      for _ in range(n_layers)]
            q = torch.randn(bs, hq, D, device="cuda", dtype=torch.bfloat16)
            indptr = torch.arange(0, bs * pages_per + 1, pages_per, device="cuda", dtype=torch.int32)
            indices = torch.arange(bs * pages_per, device="cuda", dtype=torch.int32)
            last = torch.full((bs,), kvlen - (pages_per - 1) * PAGE, device="cuda", dtype=torch.int32)
            res = {}
            for tc in (True, False):
                w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
                w.plan(indptr, indices, last, hq, hkv, D, PAGE, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)

                def run(w=w):
                    for c in caches:
                        w.run(q, c)
                try:
                    res[tc] = do_bench_cudagraph(run, rep=30) * 1e3 / n_layers
                except Exception:
                    res[tc] = do_bench(run, rep=30) * 1e3 / n_layers
            print(f"{name} bs={bs} kv={kvlen}: tensor-core {res[True]:.2f}us  cuda-core {res[False]:.2f}us", flush=True)
