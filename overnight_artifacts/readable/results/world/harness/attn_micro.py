"""LingBot self-attention shapes: FA2 varlen vs SDPA backends vs FA4 (cute). Time + diff vs FA2."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dev = torch.device("cuda")
H, D = 12, 128
res = []


def t(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


import flash_attn  # noqa: E402

try:
    from flash_attn.cute import flash_attn_func as fa4
except Exception as exc:  # noqa: BLE001
    fa4 = None
    print("no fa4", exc)
for lq, lk in ((6240, 6240), (6240, 12480), (6240, 28080), (6240, 512)):
    q = torch.randn(lq, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(lk, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(lk, H, D, device=dev, dtype=torch.bfloat16)
    cq = torch.tensor([0, lq], dtype=torch.int32, device=dev)
    ck = torch.tensor([0, lk], dtype=torch.int32, device=dev)

    def fa2():
        return flash_attn.flash_attn_varlen_func(q, k, v, cq, ck, lq, lk, 0.0, None, False)

    ref = fa2()
    row = {"lq": lq, "lk": lk, "fa2_ms": t(fa2)}
    qs, ks, vs = (x.unsqueeze(0).transpose(1, 2) for x in (q, k, v))
    for name, be in (("cudnn", SDPBackend.CUDNN_ATTENTION), ("sdpa_flash", SDPBackend.FLASH_ATTENTION),
                     ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            def f(be=be):
                with sdpa_kernel([be]):
                    return F.scaled_dot_product_attention(qs, ks, vs)
            o = f()[0].transpose(0, 1)
            row[f"{name}_ms"] = t(f)
            row[f"{name}_maxdiff"] = (o.float() - ref.float()).abs().max().item()
            row[f"{name}_exact"] = bool(torch.equal(o, ref))
        except Exception as exc:  # noqa: BLE001
            row[name] = str(exc)[:80]
    if fa4 is not None:
        try:
            def g():
                return fa4(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
            o = g()
            o = o[0] if isinstance(o, tuple) else o
            row["fa4_ms"] = t(g)
            row["fa4_maxdiff"] = (o[0].float() - ref.float()).abs().max().item()
        except Exception as exc:  # noqa: BLE001
            row["fa4"] = str(exc)[:120]
    print(row, flush=True)
