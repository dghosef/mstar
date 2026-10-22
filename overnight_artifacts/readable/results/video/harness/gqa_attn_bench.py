"""Cosmos3-Nano generation attention shape: fa3-fwd varlen vs FlashInfer ragged (fa3), GQA 32/8.

Two CFG segments, each [text prefix | gen tokens] keys, gen-token queries.
"""
import sys

import torch

gen = int(sys.argv[1]) if len(sys.argv) > 1 else 18720
text = int(sys.argv[2]) if len(sys.argv) > 2 else 128
HQ, HKV, D = (int(x) for x in (sys.argv[3:6] if len(sys.argv) > 5 else (32, 8, 128)))
B = 2
dev = "cuda"
q = torch.randn(B * gen, HQ, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B * (gen + text), HKV, D, device=dev, dtype=torch.bfloat16)
v = torch.randn_like(k)
cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * gen
cu_k = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * (gen + text)
flops = 4 * B * HQ * gen * (gen + text) * D


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


from fa3_fwd_interface import flash_attn_varlen_func  # noqa: E402

out_a = flash_attn_varlen_func(q, k, v, cu_q, cu_k, gen, gen + text, causal=False)
out_a = out_a[0] if isinstance(out_a, tuple) else out_a
t = bench(lambda: flash_attn_varlen_func(q, k, v, cu_q, cu_k, gen, gen + text, causal=False))
print(f"fa3-fwd varlen   {t:7.2f} ms  {flops / t / 1e9:5.0f} TF/s")

import flashinfer  # noqa: E402

for backend in ("fa3", "fa2"):
    ws = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=backend)
    w.plan(cu_q.cpu(), cu_k.cpu(), HQ, HKV, D, causal=False, q_data_type=torch.bfloat16)
    out_b = w.run(q, k, v)
    t = bench(lambda: w.run(q, k, v))
    err = (out_b.float() - out_a.float()).abs().max().item()
    print(f"flashinfer {backend}   {t:7.2f} ms  {flops / t / 1e9:5.0f} TF/s  maxdiff {err:.3g}")
