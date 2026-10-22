"""Time the API server's per-image preprocessing for Qwen3.5 (load_image + process_prompt)."""
import json
import sys
import time

import torch

torch.set_num_threads(4)
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel  # noqa: E402
from mstar.model.multimodal import PromptPart  # noqa: E402,F401

m = Qwen3_5DenseModel("Qwen/Qwen3.5-0.8B", cache_dir=None)
items = json.load(open("/root/work/imgset/manifest.json"))
for kind in ("small", "big", "huge"):
    paths = [it["image"] for it in items if it["kind"] == kind]
    tl, tp = [], []
    for rep in range(3):
        for p in paths:
            t0 = time.perf_counter()
            img = m.load_image(p, "cpu").data
            t1 = time.perf_counter()
            out = m.process_prompt("What is this?", ["image", "text"], ["text"], tensors={"image_inputs": [img]})
            t2 = time.perf_counter()
            if rep:
                tl.append(t1 - t0)
                tp.append(t2 - t1)
    g = out["image_grid_thw"][0].tolist()
    print(f"{kind}: load {1e3*sum(tl)/len(tl):.1f} ms  process_prompt {1e3*sum(tp)/len(tp):.1f} ms  grid {g} "
          f"pv {tuple(out['pixel_values'][0].shape)} {out['pixel_values'][0].dtype}")
print(type(m.image_processor).__name__)
sys.stdout.flush()
