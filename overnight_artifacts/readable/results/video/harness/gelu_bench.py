"""FFN up-projection + gelu(tanh): separate kernels vs cuBLASLt GELU epilogue."""
import torch
import torch.nn.functional as F

M, K, N = 36960, 3072, 14336
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
b = torch.randn(N, device="cuda", dtype=torch.bfloat16) * 0.02


def ref(x):
    return F.gelu(F.linear(x, w, b), approximate="tanh")


def epi(x):
    return torch._addmm_activation(b, x, w.t(), use_gelu=True)


def bench(fn, n=20):
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


cref = torch.compile(ref)
cepi = torch.compile(epi)
with torch.inference_mode():
    r = ref(x)
    for name, fn in [("eager ref", ref), ("compiled ref", cref), ("eager epilogue", epi), ("compiled epilogue", cepi)]:
        out = fn(x)
        d = (out.float() - r.float()).abs()
        print(f"{name:18s} {bench(fn):7.3f} ms  maxdiff {d.max().item():.3g} meandiff {d.mean().item():.3g}")
