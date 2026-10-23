import sys, time, random, torch
sys.path.insert(0, sys.argv[1])
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size="300m")
t0 = time.perf_counter(); sub = m.get_submodule("encoder", "cuda"); torch.cuda.synchronize(); print(f"load+warm {time.perf_counter()-t0:.1f}s")
random.seed(0)
groups = [[random.randint(32000, 300000) for _ in range(random.randint(2, 8))] for _ in range(60)]
with torch.no_grad():
    for label in ("first", "again"):
        torch.cuda.synchronize(); host = 0; t0 = time.perf_counter()
        for g in groups:
            if hasattr(sub, "_frontend_shape"):
                n, L = sub._frontend_shape(len(g), max(g)); lens = g + [L] * (n - len(g))
            else:
                n, L, lens = len(g), max(g), g
            x = torch.zeros(n, L, device="cuda", dtype=torch.bfloat16)
            a = time.perf_counter(); sub.model.frontend(x, lens, features=sub._compiled_features); host += time.perf_counter() - a
        torch.cuda.synchronize()
        print(f"{label}: host {1e3*host/len(groups):.2f} ms/group, total {1e3*(time.perf_counter()-t0)/len(groups):.2f}")
