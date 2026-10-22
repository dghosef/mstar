import sys, time, random, torch
sys.path.insert(0, "/root/work/head3")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size="300m")
sub = m.get_submodule("encoder", "cuda")
fe = sub._compiled_features
eager = sub.model.encoder_frontend.feature_extractor
random.seed(0)
def run(fn, shapes, label):
    torch.cuda.synchronize(); host = 0; t0 = time.perf_counter()
    for n, L in shapes:
        x = torch.zeros(n, L, device="cuda", dtype=torch.bfloat16)
        a = time.perf_counter(); fn(x); host += time.perf_counter() - a
    torch.cuda.synchronize(); tot = time.perf_counter() - t0
    print(f"{label}: host {1e3*host/len(shapes):.2f} ms/call, total {1e3*tot/len(shapes):.2f} ms/call")
with torch.no_grad():
    rnd = [(random.randint(2, 6), random.randint(32000, 300000)) for _ in range(60)]
    same = [(4, 128000)] * 60
    bucket = [(n, -(-L // 8000) * 8000) for n, L in rnd]
    for fn, name in ((fe, "compiled"), (eager, "eager")):
        run(fn, same[:5], "warm")
        run(fn, same, f"{name} same shape")
        run(fn, rnd, f"{name} random shapes")
        run(fn, rnd, f"{name} random shapes again")
        run(fn, bucket, f"{name} 0.5s buckets")
