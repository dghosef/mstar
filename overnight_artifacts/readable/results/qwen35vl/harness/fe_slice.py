import sys, time, random, torch
import torch.nn.functional as F
sys.path.insert(0, "/root/work/head3")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size="300m"); sub = m.get_submodule("encoder", "cuda")
fe = sub.model.encoder_frontend.feature_extractor
def layer_slices(layer, x):
    conv = layer.conv; (k,), (s,) = conv.kernel_size, conv.stride
    t = (x.shape[1] - k) // s + 1
    w = conv.weight  # (Cout, Cin, k)
    y = None
    for j in range(k):
        part = x[:, j: j + s * (t - 1) + 1: s] @ w[:, :, j].t()
        y = part if y is None else y + part
    if conv.bias is not None: y = y + conv.bias
    ln = layer.layer_norm
    y = F.layer_norm(y.float(), ln.normalized_shape, ln.weight.float(), ln.bias.float(), ln.eps).type(y.dtype)
    return F.gelu(y)
def fe_slices(wv):
    x = wv.unsqueeze(-1)
    for layer in fe.layers: x = layer_slices(layer, x)
    return x
c = torch.compile(fe_slices, dynamic=True)
fec = sub._compiled_features
random.seed(0)
shapes = [(random.randint(1, 8), random.randint(32000, 300000)) for _ in range(40)]
with torch.no_grad():
    c(torch.zeros(1, 16000, device="cuda", dtype=torch.bfloat16)); c(torch.zeros(2, 16000, device="cuda", dtype=torch.bfloat16))
    x = torch.randn(4, 160000, device="cuda", dtype=torch.bfloat16)
    a, b = fec(x), c(x)
    print("diff", (a.float() - b.float()).abs().max().item(), ((a.float()-b.float()).norm()/a.float().norm()).item())
    for fn, name in ((fec, "conv"), (c, "slices")):
        for rep in range(2):
            torch.cuda.synchronize(); host = 0; t0 = time.perf_counter()
            for n, L in shapes:
                xx = torch.zeros(n, L, device="cuda", dtype=torch.bfloat16)
                t1 = time.perf_counter(); fn(xx); host += time.perf_counter() - t1
            torch.cuda.synchronize()
            print(f"{name} pass{rep}: host {1e3*host/len(shapes):.2f} ms, total {1e3*(time.perf_counter()-t0)/len(shapes):.2f} ms")
