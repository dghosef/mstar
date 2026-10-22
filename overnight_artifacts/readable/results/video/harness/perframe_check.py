"""Wan2.2 DiT: per-token vs per-frame timestep, same values. Eager exactness + compiled timing."""
import time

import torch

from mstar.model.wan22.config import Wan22Config
from mstar.model.wan22.weight_loader import build_wan22_dit

dev = torch.device("cuda")
dit = build_wan22_dit(Wan22Config(), "Wan-AI/Wan2.2-TI2V-5B-Diffusers", device=dev)
B, F, H, W = 2, 21, 44, 80
S = F * (H // 2) * (W // 2)
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(B, 48, F, H, W, device=dev, dtype=torch.bfloat16, generator=g)
txt = torch.randn(B, 512, 4096, device=dev, dtype=torch.bfloat16, generator=g)
frame_t = torch.full((B, F), 777.0, device=dev)
frame_t[:, 0] = 0.0  # i2v-like first frame
token_t = frame_t.repeat_interleave(S // F, dim=1)

with torch.inference_mode():
    a = dit(x, token_t, txt)
    b = dit(x, frame_t, txt)
    d = (a.float() - b.float()).abs()
    print(f"eager per-token vs per-frame: max {d.max().item():.4g} mean {d.mean().item():.4g} exact={torch.equal(a, b)}")
    dit.set_fuse_ffn_gelu(True)
    fwd = torch.compile(dit.forward, fullgraph=False, dynamic=False)
    for name, ts in (("per-token", token_t), ("per-frame", frame_t)):
        for _ in range(3):
            fwd(x, ts, txt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            fwd(x, ts, txt)
        torch.cuda.synchronize()
        print(f"compiled {name}: {(time.perf_counter() - t0) / 5 * 1000:.1f} ms/forward (sdpa)")
