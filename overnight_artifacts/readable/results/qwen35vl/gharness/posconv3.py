import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size=sys.argv[2]); sub = m.get_submodule("encoder", "cuda")
pe = sub.model.encoder_frontend.pos_encoder
conv = pe.conv
def v2(x):
    n, t, c = x.shape
    groups, (kernel,), (pad,) = conv.groups, conv.kernel_size, conv.padding
    cg = c // groups
    xg = F.pad(x, (0, 0, pad, pad)).view(n, t + 2 * pad, groups, cg).permute(2, 0, 1, 3).contiguous()  # (G, N, Tp, Cg)
    t_out = t + 2 * pad - kernel + 1
    win = xg.unfold(2, kernel, 1)  # (G, N, T', Cg, K)
    w = conv.weight.reshape(groups, cg, cg * kernel).transpose(1, 2)
    y = torch.bmm(win.reshape(groups, n * t_out, cg * kernel), w)  # (G, N*T', Cg)
    return y.view(groups, n, t_out, cg).permute(1, 2, 0, 3).reshape(n, t_out, c) + conv.bias
def bench(fn, *a, n=5):
    for _ in range(2): fn(*a)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn(*a)
    torch.cuda.synchronize(); return 1e3 * (time.perf_counter() - t0) / n
with torch.no_grad():
    for N, T in ((1, 370), (4, 500), (8, 600), (16, 1000)):
        x = torch.randn(N, T, conv.in_channels, device="cuda", dtype=conv.weight.dtype)
        a = pe._conv_unfolded(x); b = v2(x)
        print(N, T, f"v1 {bench(pe._conv_unfolded, x):.2f} ms  v2 {bench(v2, x):.2f} ms  cudnn {bench(lambda z: conv(z.transpose(1, 2)), x):.2f} ms  maxdiff {(a.float()-b.float()).abs().max().item():.4f}")
