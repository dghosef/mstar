"""Feature extractor as unfold+GEMM (channels-last) vs cudnn conv: equality and host/GPU time over random shapes."""
import sys, time, random, torch
import torch.nn.functional as F
sys.path.insert(0, "/root/work/head3")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size="300m")
sub = m.get_submodule("encoder", "cuda")
fe = sub.model.encoder_frontend.feature_extractor

def fe_gemm(w):  # (N, S) -> (N, T, C)
    x = w.unsqueeze(-1)  # (N, S, 1) channels-last
    for layer in fe.layers:
        conv = layer.conv
        k, s = conv.kernel_size[0], conv.stride[0]
        cin, cout = conv.in_channels, conv.out_channels
        win = x.unfold(1, k, s)  # (N, T', C, k)
        n, t = win.shape[:2]
        y = win.reshape(n, t, cin * k) @ conv.weight.reshape(cout, cin * k).t()
        if conv.bias is not None:
            y = y + conv.bias
        ln = layer.layer_norm
        y = F.layer_norm(y.float(), ln.normalized_shape, ln.weight.float(), ln.bias.float(), ln.eps).type(y.dtype)
        x = F.gelu(y)
    return x

fe_gemm_c = torch.compile(fe_gemm, dynamic=True)
fec = sub._compiled_features
random.seed(0)
shapes = [(random.randint(2, 8), random.randint(32000, 300000)) for _ in range(40)]
with torch.no_grad():
    x = torch.randn(4, 160000, device="cuda", dtype=torch.bfloat16)
    a = fec(x); b = fe_gemm_c(x); c = fe(x)
    print("shape", tuple(a.shape), tuple(b.shape), "max|gemm-conv_compiled|", (a.float() - b.float()).abs().max().item(),
          "max|conv_eager-conv_compiled|", (a.float() - c.float()).abs().max().item(), "rel", ((a.float()-b.float()).norm()/a.float().norm()).item())
    for fn, name in ((fec, "conv compiled"), (fe_gemm_c, "gemm compiled")):
        for rep in range(2):
            torch.cuda.synchronize(); host = 0; t0 = time.perf_counter()
            for n, L in shapes:
                xx = torch.zeros(n, L, device="cuda", dtype=torch.bfloat16)
                t1 = time.perf_counter(); fn(xx); host += time.perf_counter() - t1
            torch.cuda.synchronize()
            print(f"{name} pass{rep}: host {1e3*host/len(shapes):.2f} ms, total {1e3*(time.perf_counter()-t0)/len(shapes):.2f} ms")
    LAD = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40)
    def bucket(L):
        for s in LAD:
            if L <= s * 16000: return s * 16000
        return L
    bshapes = [(n, bucket(L)) for n, L in shapes]
    print("distinct bucketed shapes", len(set(bshapes)), "of", len(bshapes))
    for fn, name in ((fec, "conv compiled bucketed"), (fe_gemm_c, "gemm compiled bucketed")):
        for rep in range(2):
            torch.cuda.synchronize(); host = 0; t0 = time.perf_counter()
            for n, L in bshapes:
                xx = torch.zeros(n, L, device="cuda", dtype=torch.bfloat16)
                t1 = time.perf_counter(); fn(xx); host += time.perf_counter() - t1
            torch.cuda.synchronize()
            print(f"{name} pass{rep}: host {1e3*host/len(shapes):.2f} ms, total {1e3*(time.perf_counter()-t0)/len(shapes):.2f} ms")
