import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, "/root/work/head4")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
size = sys.argv[1]
m = OmnilingualASRModel(size=size); sub = m.get_submodule("encoder", "cuda")
pe = sub.model.encoder_frontend.pos_encoder
conv = pe.conv
C = conv.in_channels; G = conv.groups; K = conv.kernel_size[0]; P = conv.padding[0]
print("C", C, "G", G, "K", K, "pad", P, conv.weight.dtype)
def bench(fn, *a, n=10):
    for _ in range(3): fn(*a)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn(*a)
    torch.cuda.synchronize(); return 1e3 * (time.perf_counter() - t0) / n
def nchw(x):  # x (N, C, T)
    return conv(x)
def cl2d(x):
    x4 = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)
    w4 = conv.weight.unsqueeze(2).contiguous(memory_format=torch.channels_last)
    return F.conv2d(x4, w4, conv.bias, padding=(0, P), groups=G).squeeze(2)
def taps(x):  # per-tap batched GEMM over groups: y[g] = sum_k W[g,:,:,k] @ x_pad[g, :, k:k+Tout]
    N, _, T = x.shape
    xp = F.pad(x, (P, P)).view(N, G, C // G, T + 2 * P)
    Tout = T + 2 * P - K + 1
    w = conv.weight.view(G, C // G, C // G, K)
    y = None
    for k in range(K):
        part = torch.einsum("goi,ngit->ngot", w[..., k], xp[..., k:k + Tout])
        y = part if y is None else y + part
    return (y.reshape(N, C, Tout) + conv.bias[None, :, None])
def unfold_bmm(x):
    N, _, T = x.shape
    xp = F.pad(x, (P, P))
    Tout = T + 2 * P - K + 1
    win = xp.unfold(2, K, 1)  # (N, C, Tout, K)
    win = win.view(N, G, C // G, Tout, K).permute(0, 1, 3, 2, 4).reshape(N * G, Tout, (C // G) * K)
    w = conv.weight.view(G, C // G, (C // G) * K)  # (G, O, I*K)
    y = torch.bmm(win, w.repeat(N, 1, 1).transpose(1, 2))  # (N*G, Tout, O)
    return y.view(N, G, Tout, C // G).permute(0, 1, 3, 2).reshape(N, C, Tout) + conv.bias[None, :, None]
with torch.no_grad():
    for N, T in ((1, 370), (8, 600), (16, 950)):
        x = torch.randn(N, C, T, device="cuda", dtype=conv.weight.dtype)
        ref = nchw(x)
        res = {"nchw": bench(nchw, x)}
        for name, fn in (("cl2d", cl2d), ("unfold_bmm", unfold_bmm)):
            try:
                y = fn(x); err = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
                res[name] = f"{bench(fn, x):.2f} ms (rel {err:.1e})"
            except Exception as e:
                res[name] = f"ERR {type(e).__name__}: {str(e)[:60]}"
        print(N, T, {k: (f"{v:.2f} ms" if isinstance(v, float) else v) for k, v in res.items()})
    print("--- benchmark mode / cudnn off")
    for N, T in ((1, 370), (8, 600), (16, 950)):
        x = torch.randn(N, C, T, device="cuda", dtype=conv.weight.dtype)
        torch.backends.cudnn.benchmark = True
        t0 = time.perf_counter(); conv(x); torch.cuda.synchronize(); first = 1e3 * (time.perf_counter() - t0)
        b = bench(nchw, x)
        torch.backends.cudnn.benchmark = False
        with torch.backends.cudnn.flags(enabled=False):
            try:
                off = f"{bench(nchw, x):.2f} ms"
            except Exception as e:
                off = f"ERR {e}"
        print(N, T, f"benchmark first call {first:.0f} ms then {b:.2f} ms; cudnn off {off}")
