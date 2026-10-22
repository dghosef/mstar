import sys, time, torch
sys.path.insert(0, sys.argv[1])
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size=sys.argv[2]); sub = m.get_submodule("encoder", "cuda")
pe = sub.model.encoder_frontend.pos_encoder
def bench(fn, *a, n=5):
    for _ in range(2): fn(*a)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn(*a)
    torch.cuda.synchronize(); return 1e3 * (time.perf_counter() - t0) / n
C = pe.conv.in_channels
with torch.no_grad():
    for N, T in ((1, 370), (1, 1999), (4, 500), (8, 600), (16, 950), (16, 1000)):
        x = torch.randn(N, T, C, device="cuda", dtype=pe.conv.weight.dtype)
        mask = torch.ones(N, T, dtype=torch.bool, device="cuda"); mask[0, T // 2:] = False
        pe.unfolded = False; ref = pe(x, mask); tr = bench(pe, x, mask)
        pe.unfolded = True; got = pe(x, mask); tu = bench(pe, x, mask)
        err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"N={N} T={T}: cudnn {tr:.2f} ms, unfolded {tu:.2f} ms, rel err {err:.1e}, maxabs {(got.float()-ref.float()).abs().max().item():.3f}")
