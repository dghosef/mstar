"""LingBot DiT: time one chunk's forwards (5 per chunk) and profile ops, real weights,
random-but-realistic-shape inputs; also dumps the flow output for cross-arm comparison.

  python lb_dit_micro.py --arm /root/work/mstar --tag new --out /workspace/results/world/lb_dit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chunks", type=int, default=6)
ap.add_argument("--profile", action="store_true")
args = ap.parse_args()
sys.path.insert(0, args.arm)
from mstar.model.lingbot_world.config import lingbot_world_1_3b  # noqa: E402
from mstar.model.lingbot_world.weight_loader import build_lingbot_dit  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cfg = lingbot_world_1_3b()
dev = torch.device("cuda")
dit = build_lingbot_dit(cfg, "robbyant/lingbot-world-v2-1.3b-causal-fast", device="cuda")
lat_f, lat_h, lat_w = 4 * args.chunks, 60, 104
frame_seqlen = lat_h * lat_w // 4
g = torch.Generator(device=dev).manual_seed(0)
noise = torch.randn(16, lat_f, lat_h, lat_w, generator=g, device=dev)
y = torch.randn(20, lat_f, lat_h, lat_w, generator=g, device=dev)
plucker = torch.randn(1, 384, lat_f, lat_h, lat_w, generator=g, device=dev).to(torch.bfloat16)
ctx = torch.randn(512, 4096, generator=g, device=dev).to(torch.bfloat16)
ts = [torch.tensor([t], device=dev, dtype=torch.float32) for t in (1000.0, 750.0, 500.0, 250.0, 0.0)]


def rollout():
    kv = dit.new_cache(frame_seqlen, lat_f, dev)
    outs = []
    for c in range(args.chunks):
        fr = slice(4 * c, 4 * c + 4)
        pl = plucker[:, :, fr]
        for t in ts:
            o = dit(noise[:, fr], t, ctx, y[:, fr], pl, kv, c * 4 * frame_seqlen, frame_seqlen)
        outs.append(o)
    return outs


with torch.inference_mode():
    rollout()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = rollout()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
res = {"tag": args.tag, "ms_per_chunk": 1e3 * dt / args.chunks, "ms_per_forward": 1e3 * dt / (5 * args.chunks)}
o = torch.stack([x.float().cpu() for x in outs])
torch.save(o, out / f"flow_{args.tag}.pt")
ref_p = out / "flow_base.pt"
if args.tag != "base" and ref_p.exists():
    ref = torch.load(ref_p)
    res["bit_exact_vs_base"] = bool(torch.equal(o, ref))
    res["max_abs_vs_base"] = (o - ref).abs().max().item()
    res["cos_vs_base"] = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
print(json.dumps(res))
(out / f"res_{args.tag}.json").write_text(json.dumps(res, indent=1))
if args.profile:
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
        kv = dit.new_cache(frame_seqlen, lat_f, dev)
        pl = plucker[:, :, 4:8]
        for t in ts:
            dit(noise[:, 4:8], t, ctx, y[:, 4:8], pl, kv, 4 * frame_seqlen, frame_seqlen)
        torch.cuda.synchronize()
    tbl = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=60, max_name_column_width=40, max_shapes_column_width=80)
    (out / f"prof_{args.tag}.txt").write_text(tbl)
    print(tbl[:12000])
