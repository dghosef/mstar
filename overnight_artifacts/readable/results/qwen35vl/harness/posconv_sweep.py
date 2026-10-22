import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, "/root/work/head4")
exec(open("/root/work/vlh/posconv.py").read().split("with torch.no_grad():")[0].split("print(\"C\"")[0])
def bench(fn, *a, n=5):
    for _ in range(2): fn(*a)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn(*a)
    torch.cuda.synchronize(); return 1e3 * (time.perf_counter() - t0) / n
conv = sub.model.encoder_frontend.pos_encoder.conv
with torch.no_grad():
    for N in (1, 2, 4, 8, 12, 16):
        row = []
        for T in (100, 250, 500, 1000, 2000):
            if N * T > 16000: row.append("-"); continue
            x = torch.randn(N, conv.in_channels, T, device="cuda", dtype=conv.weight.dtype)
            row.append(f"{bench(conv, x):.1f}")
        print("cudnn N=%2d" % N, row)
