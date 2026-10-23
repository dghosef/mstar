#!/usr/bin/env python3
"""Offline competitor for Waypoint-1.5: Overworld's official world_engine (tag 1.5.6, what Overworld's Biome
server ships), bf16, no quantization, float32 matmul precision 'high' (as M* serves), its own compile +
CUDA graphs. One world per engine (StaticKVCache batch 1, no batching), so "batch b" = b processes on the
same GPU, each running its share of the rollouts; the throughput is total frames / wall from a common start.
Per rollout: reset, prime on the seed image (4 copies, like M*), the same 64-step action script, decode +
.cpu() of every [4, H, W, 3] chunk (what M*'s client receives). Saves the last chunk (uint8 RGB) for the gate.
"""
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doff_common as D  # noqa: E402


def extra(ap):
    ap.add_argument("--model", required=True)


def worker(model, item, n, warm, start_evt, q, save_last):
    import numpy as np
    import torch
    from PIL import Image
    from world_engine import CtrlInput, WorldEngine
    torch.set_float32_matmul_precision("high")
    eng = WorldEngine(model, quant=None, device="cuda", dtype=torch.bfloat16)
    img = np.asarray(Image.open(item["image_path"]).convert("RGB"))
    seed4 = torch.from_numpy(np.repeat(img[None], 4, axis=0)).cuda()
    mk = json.loads(item["model_kwargs"])

    def rollout():
        eng.reset()
        torch.manual_seed(mk["seed"])
        eng.append_frame(seed4)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first, fr = None, None
        for a in mk["actions"]:
            c = CtrlInput(button=set(a["buttons"]), mouse=tuple(a["mouse"]), scroll_wheel=int(a["scroll"]))
            fr = eng.gen_frame(ctrl=c).cpu()
            if first is None:
                first = time.perf_counter() - t0
        return time.perf_counter() - t0, first, fr

    for _ in range(max(2, warm)):
        rollout()
    q.put(("ready", None))
    start_evt.wait()
    res, last = [], None
    for _ in range(n):
        dt, first, last = rollout()
        res.append((dt, first, time.time()))
    q.put(("done", {"res": res, "last": last.numpy().tobytes() if save_last else None}))


def main():
    a = D.parse(extra)
    item = a.items_list[0]
    rows = []
    for b, n in zip([int(x) for x in a.batch_sizes.split(",")], [int(x) for x in a.nums.split(",")]):
        ctx = mp.get_context("spawn")
        q, evt = ctx.Queue(), ctx.Event()
        per = [n // b + (1 if i < n % b else 0) for i in range(b)]
        ps = [ctx.Process(target=worker, args=(a.model, item, per[i], a.warmup, evt, q, i == 0 and bool(a.save_dir)))
              for i in range(b) if per[i]]
        for p in ps:
            p.start()
        try:
            for _ in ps:
                kind, _ = q.get(timeout=3600)
            t0 = time.time()
            evt.set()
            outs = [q.get(timeout=7200)[1] for _ in ps]
            wall = max(r[2] for o in outs for r in o["res"]) - t0
            lat = [r[0] for o in outs for r in o["res"]]
            ttfb = [r[1] for o in outs for r in o["res"]]
            done = len(lat)
            rows.append({"batch": b, "num": n, "completed": done, "failed": n - done, "wall_s": round(wall, 3),
                         "throughput_per_s": done / wall, "throughput_per_min": 60 * done / wall,
                         "units_per_s": done * item["units"] / wall, "units_total": done * item["units"],
                         "lat_mean": statistics.fmean(lat), "lat_p50": D.pct(lat, 50), "lat_p90": D.pct(lat, 90),
                         "lat_p99": D.pct(lat, 99), "ttfb_p50": D.pct(ttfb, 50), "ttfb_p99": D.pct(ttfb, 99),
                         "processes": len(ps), "errors": []})
            last = next((o["last"] for o in outs if o.get("last")), None)
            if a.save_dir and last:
                for i in range(n):
                    (Path(a.save_dir) / f"{i:03d}.bin").write_bytes(last)
        except Exception as e:  # noqa: BLE001
            rows.append({"batch": b, "num": n, "completed": 0, "failed": None, "error": repr(e)[:500]})
        finally:
            for p in ps:
                p.join(timeout=60)
                if p.is_alive():
                    p.kill()
        print(json.dumps(rows[-1]), flush=True)
        (Path(a.out) / "results.json").write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
