import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size=sys.argv[2]); sub = m.get_submodule("encoder", "cuda")
pe = sub.model.encoder_frontend.pos_encoder; conv = pe.conv
def taps(x):
    n, t, c = x.shape
    groups, (kernel,), (pad,) = conv.groups, conv.kernel_size, conv.padding
    cg = c // groups
    tp = t + 2 * pad
    xg = F.pad(x, (0, 0, pad, pad)).view(n, tp, groups, cg).permute(2, 0, 1, 3).contiguous().float()  # (G,N,Tp,Cg)
    t_out = tp - kernel + 1
    w = conv.weight.float().view(groups, cg, cg, kernel)  # (G, O, I, K)
    wk = w.permute(3, 0, 2, 1).contiguous()  # (K, G, I, O)
    out = torch.zeros(groups * n, t_out, cg, device=x.device, dtype=torch.float32)
    base = xg.view(groups * n, tp, cg)
    for k in range(kernel):
        xk = base[:, k:k + t_out]  # (G*N, T', I) strided view
        wkk = wk[k].unsqueeze(1).expand(groups, n, cg, cg).reshape(groups * n, cg, cg)
        out.baddbmm_(xk, wkk)
    y = out.view(groups, n, t_out, cg).permute(1, 2, 0, 3).reshape(n, t_out, c).to(x.dtype)
    return y + conv.bias
def bench(fn, *a, n=5):
    for _ in range(2): fn(*a)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn(*a)
    torch.cuda.synchronize(); return 1e3 * (time.perf_counter() - t0) / n
torch.backends.cuda.matmul.allow_tf32 = False
with torch.no_grad():
    for N, T in ((1, 370), (8, 600), (16, 1000)):
        x = torch.randn(N, T, conv.in_channels, device="cuda", dtype=conv.weight.dtype)
        a = pe._conv_unfolded(x); b = taps(x)
        print(N, T, f"unfold {bench(pe._conv_unfolded, x):.2f} ms  taps-fp32 {bench(taps, x):.2f} ms  maxdiff {(a.float()-b.float()).abs().max().item():.4f}")
