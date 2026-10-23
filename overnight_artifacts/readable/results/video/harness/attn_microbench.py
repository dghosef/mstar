"""Wan2.2 DiT attention shapes: SDPA backends vs FlashInfer ragged (fa2/fa3).

python attn_microbench.py [--tokens 18480] [--batch 2]
"""
import argparse

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, default=21 * 22 * 40)   # 1280x704x81 post-patch
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--heads", type=int, default=24)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--text", type=int, default=512)
args = ap.parse_args()
dev = "cuda"
B, S, H, D, T = args.batch, args.tokens, args.heads, args.dim, args.text
q = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)
kt = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
vt = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


flops_self = 4 * B * H * S * S * D
flops_x = 4 * B * H * S * T * D


def sdpa(qq, kk, vv):
    return F.scaled_dot_product_attention(qq.permute(0, 2, 1, 3), kk.permute(0, 2, 1, 3),
                                          vv.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)


ref = sdpa(q, k, v)
ref_x = sdpa(q, kt, vt)
print(f"torch {torch.__version__} cudnn {torch.backends.cudnn.version()} {torch.cuda.get_device_name()}")
print(f"B={B} S={S} H={H} D={D}")
for name, be in [("default", None), ("flash", SDPBackend.FLASH_ATTENTION),
                 ("cudnn", SDPBackend.CUDNN_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)]:
    try:
        if be is None:
            t = bench(lambda: sdpa(q, k, v))
            tx = bench(lambda: sdpa(q, kt, vt))
        else:
            with sdpa_kernel([be]):
                t = bench(lambda: sdpa(q, k, v))
                tx = bench(lambda: sdpa(q, kt, vt))
        print(f"sdpa[{name:9s}] self {t:7.2f} ms ({flops_self / t / 1e9:6.0f} TF/s)  "
              f"cross {tx:6.3f} ms ({flops_x / tx / 1e9:6.0f} TF/s)")
    except Exception as ex:  # noqa: BLE001
        print(f"sdpa[{name}] failed: {ex!r}"[:200])

import flashinfer  # noqa: E402

for backend in ("fa2", "fa3"):
    try:
        ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=backend)
        cu = torch.arange(0, B + 1, dtype=torch.int32) * S
        w.plan(cu, cu, H, H, D, causal=False, q_data_type=torch.bfloat16)
        qf, kf, vf = q.reshape(B * S, H, D), k.reshape(B * S, H, D), v.reshape(B * S, H, D)
        t = bench(lambda: w.run(qf, kf, vf))
        out = w.run(qf, kf, vf).view(B, S, H, D)
        err = (out.float() - ref.float()).abs().max().item()
        ws2 = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        w2 = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws2, "NHD", backend=backend)
        cuk = torch.arange(0, B + 1, dtype=torch.int32) * T
        w2.plan(cu, cuk, H, H, D, causal=False, q_data_type=torch.bfloat16)
        kft, vft = kt.reshape(B * T, H, D), vt.reshape(B * T, H, D)
        tx = bench(lambda: w2.run(qf, kft, vft))
        errx = (w2.run(qf, kft, vft).view(B, S, H, D).float() - ref_x.float()).abs().max().item()
        print(f"flashinfer[{backend}] self {t:7.2f} ms ({flops_self / t / 1e9:6.0f} TF/s) maxerr {err:.3g}  "
              f"cross {tx:6.3f} ms ({flops_x / tx / 1e9:6.0f} TF/s) maxerr {errx:.3g}")
    except Exception as ex:  # noqa: BLE001
        print(f"flashinfer[{backend}] failed: {ex!r}"[:300])

try:
    import flash_attn_interface as fa3  # noqa: E402
    t = bench(lambda: fa3.flash_attn_func(q, k, v))
    print(f"fa3 wheel self {t:7.2f} ms ({flops_self / t / 1e9:6.0f} TF/s)")
except Exception as ex:  # noqa: BLE001
    print(f"fa3 wheel: {ex!r}"[:200])
