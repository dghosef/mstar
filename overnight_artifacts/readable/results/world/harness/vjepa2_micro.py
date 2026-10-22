"""V-JEPA2 encoder (+ masked predictor) timing and output dump, real weights.

Run once per arm with that arm's tree first on sys.path:
  PYTHONPATH=/root/work/base python vjepa2_micro.py --tag base --out /workspace/results/vjepa2_micro
  PYTHONPATH=/root/work/mstar python vjepa2_micro.py --tag new  --out /workspace/results/vjepa2_micro
The second run compares against the first's dumped outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="vjepa2", choices=("vjepa2", "vjepa2_ac"))
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--compile", action="store_true")
ap.add_argument("--frames", type=int, default=0, help="frames per clip (0: config frames_per_clip)")
args = ap.parse_args()

import mstar  # noqa: E402
from mstar.model.registry import get_model_class, model_init_kwargs  # noqa: E402

dev, dt = torch.device("cuda"), torch.bfloat16
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

model = get_model_class(args.model)(**model_init_kwargs(args.model))
enc_sub = model.get_submodule("video_encoder", device="cuda", autocast_dtype=dt)
enc = enc_sub.encoder.eval()
cfg = model.config


def time_ms(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


fwd = torch.compile(enc) if args.compile else enc
res = {"tag": args.tag, "mstar": mstar.__file__, "model": args.model, "compile": args.compile,
       "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "enc": {}}
g = torch.Generator(device="cuda").manual_seed(0)
x1 = torch.randn(1, args.frames or cfg.frames_per_clip, 3, cfg.crop_size, cfg.crop_size, device=dev, dtype=dt, generator=g)
with torch.inference_mode():
    for b in args.batch:
        x = x1.expand(b, -1, -1, -1, -1).contiguous()
        torch.cuda.reset_peak_memory_stats()
        try:
            ms = time_ms(lambda: fwd(x), args.iters)
            res["enc"][str(b)] = {"ms": ms, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        except torch.OutOfMemoryError as exc:
            res["enc"][str(b)] = {"oom": str(exc)[:200]}
        torch.cuda.empty_cache()
    out = fwd(x1).float().cpu()

sfx = f"f{args.frames}" if args.frames else ""
tagged = out_dir / f"{args.model}{sfx}_{'c' if args.compile else 'e'}_{args.tag}_enc.pt"
torch.save(out, tagged)
base = out_dir / f"{args.model}{sfx}_{'c' if args.compile else 'e'}_base_enc.pt"
if args.tag != "base" and base.exists():
    ref = torch.load(base)
    res["cos_vs_base"] = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    res["max_abs_vs_base"] = (out - ref).abs().max().item()
    res["ref_abs_mean"] = ref.abs().mean().item()
    res["per_token_cos_min"] = torch.nn.functional.cosine_similarity(out[0], ref[0], dim=-1).min().item()
(out_dir / f"{args.model}{sfx}_{'c' if args.compile else 'e'}_{args.tag}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
