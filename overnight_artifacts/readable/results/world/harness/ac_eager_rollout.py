"""Eager AC rollout (full-sequence attention, no KV cache) at B=1 vs B=4 identical rows,
and its sensitivity to a tiny input perturbation: is batch variance just rounding amplification?"""
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, sys.argv[1])
from mstar.model.registry import get_model_class, model_init_kwargs  # noqa: E402

m = get_model_class("vjepa2_ac")(**model_init_kwargs("vjepa2_ac"))
pred = m.get_submodule("rollout_predictor", device="cuda", autocast_dtype=torch.bfloat16).predictor
kw = json.load(open(sys.argv[2]))
act = torch.tensor(kw["actions"], device="cuda")[None]
st = torch.tensor(kw["states"], device="cuda")[None]
g = torch.Generator(device="cuda").manual_seed(0)
ctx0 = F.layer_norm(torch.randn(1, 256, 1408, device="cuda", generator=g), (1408,)).to(torch.bfloat16)


def rollout(ctx, b):
    ctx = ctx.expand(b, -1, -1).contiguous()
    outs = []
    for k in range(4):
        y = pred(ctx, act[:, : k + 1].expand(b, -1, -1).to(ctx.dtype), st[:, : k + 1].expand(b, -1, -1).to(ctx.dtype))
        new = F.layer_norm(y[:, -256:].float(), (1408,)).to(ctx.dtype)
        outs.append(new.float())
        ctx = torch.cat([ctx, new], dim=1)
    return torch.stack(outs, 1)  # [b, 4, 256, D]


r = lambda a, b: ((a - b).norm() / a.norm()).item()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    o1 = rollout(ctx0, 1)
    o4 = rollout(ctx0, 4)
    print("B=4 rows vs B=1 per step:", [[round(r(o1[0, k], o4[i, k]), 4) for k in range(4)] for i in range(4)])
    for eps in (1e-4, 1e-3):
        op = rollout((ctx0.float() + eps * torch.randn_like(ctx0.float())).to(ctx0.dtype), 1)
        print(f"input noise {eps}: per step", [round(r(o1[0, k], op[0, k]), 4) for k in range(4)])
