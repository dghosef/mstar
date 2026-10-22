"""A/B a Cosmos3 DROID policy server (action loop over /generate/ws) across arms.

  python ab_cosmos.py --model cosmos3_droid --arms base=/root/work/base new=/root/work/mstar --out DIR
Per arm: fresh server, then for each --pipeline value one ws session of --iters chunks
(first --warmup excluded); actions of every chunk saved (npy) for an output comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets.sync.client

sys.path.insert(0, str(Path(__file__).parent))
from srv import Server, phases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="cosmos3_droid")
ap.add_argument("--config", default=None)
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--pipelines", type=int, nargs="+", default=[1, 4])
ap.add_argument("--iters", type=int, default=24)
ap.add_argument("--warmup", type=int, default=4)
ap.add_argument("--image", default="/root/work/media/seed.jpg")
ap.add_argument("--prompt", default="Pick up the red cup and place it in the sink.")
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--action-dim", type=int, default=10)
ap.add_argument("--port", type=int, default=9140)
ap.add_argument("--gpus", default="0")
ap.add_argument("--period", default="20")
ap.add_argument("--out", required=True)
ap.add_argument("--env", nargs="*", default=[])
args = ap.parse_args()
arms = dict(a.split("=", 1) for a in args.arms)
extra: dict[str, dict] = {}
for e in args.env:
    arm, kv = e.split(":", 1)
    k, v = kv.split("=", 1)
    extra.setdefault(arm, {})[k] = v
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
obs = Path(args.image).read_bytes()
mk = {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": args.action_dim,
      "action_chunk_size": args.chunk, "num_frames": args.chunk + 1, "seed": 0}


def session(pipeline: int, tag: str):
    def msg(i):
        return msgpack.packb({"text": args.prompt, "files": [{"name": f"obs_{i}.jpg", "data": obs}],
                              "input_modalities": ["image", "text"], "output_modalities": ["action"],
                              "model_kwargs": mk, "request_id": f"{tag}-{i}"}, use_bin_type=True)
    lat, fin, acts, sent = [], [], {}, {}
    with websockets.sync.client.connect(f"ws://127.0.0.1:{args.port}/generate/ws", max_size=None) as ws:
        t0 = time.perf_counter()
        nxt = done = 0
        while nxt < min(pipeline, args.iters):
            sent[f"{tag}-{nxt}"] = time.perf_counter()
            ws.send(msg(nxt))
            nxt += 1
        while done < args.iters:
            r = msgpack.unpackb(ws.recv(), raw=False)
            if "error" in r:
                raise RuntimeError(r["error"])
            if r.get("modality") == "action":
                acts[r["request_id"]] = np.frombuffer(r["data"], dtype=np.float32).reshape(args.chunk, -1).copy()
                lat.append(time.perf_counter() - sent[r["request_id"]])
            if r.get("finish"):
                done += 1
                fin.append(time.perf_counter())
                if nxt < args.iters:
                    sent[f"{tag}-{nxt}"] = time.perf_counter()
                    ws.send(msg(nxt))
                    nxt += 1
    w = args.warmup
    steady = lat[w:]
    wall = fin[-1] - fin[w - 1]
    return {"pipeline": pipeline, "chunks_per_s": len(steady) / wall, "lat_p50_ms": 1e3 * statistics.median(steady),
            "lat_p90_ms": 1e3 * sorted(steady)[int(0.9 * (len(steady) - 1))], "first_ms": 1e3 * lat[0]}, acts


summary = []
for rnd in range(args.rounds):
    order = list(arms) if rnd % 2 == 0 else list(arms)[::-1]
    for name in order:
        d = out / f"r{rnd}_{name}"
        d.mkdir(parents=True, exist_ok=True)
        log = str(d / "server.log")
        srv = Server(arms[name], [args.model, "--config", args.config or f"configs/{args.model}.yaml"], args.port,
                     log, gpus=args.gpus, env={"MSTAR_PHASE_TIMING": args.period, **extra.get(name, {})},
                     timeout=3600)
        startup = srv.start()
        try:
            for p in args.pipelines:
                res, acts = session(p, f"p{p}")
                res.update(round=rnd, arm=name, startup_s=startup)
                summary.append(res)
                np.save(d / f"actions_p{p}.npy", np.stack([acts[k] for k in sorted(acts, key=lambda s: int(s.split('-')[1]))]))
                print(json.dumps(res), flush=True)
        finally:
            srv.stop()
        try:
            (d / "phases.txt").write_text(phases(log))
        except Exception as exc:  # noqa: BLE001
            (d / "phases.txt").write_text(f"phase parse failed: {exc}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
print("AB DONE")
