"""A/B LingBot-World 1.3B across arms: one server per arm per round, a warmup rollout,
then measured rollouts with per-chunk arrival times; frames saved (npy, box only).

  python ab_lingbot.py --arms base=/root/guest-world/base new=/root/guest-world/mstar \
      --rounds 2 --out /workspace/results/world/lingbot_ab --num-frames 93
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from srv import Server, phases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--num-frames", type=int, default=93)
ap.add_argument("--example", default="/root/work/lingbot-world/examples/03")
ap.add_argument("--port", type=int, default=9120)
ap.add_argument("--gpus", default="0")
ap.add_argument("--period", default="2")
ap.add_argument("--out", required=True)
ap.add_argument("--env", nargs="*", default=[], help="arm:KEY=VAL extra env per arm")
ap.add_argument("--configs", nargs="*", default=[], help="arm:configs/x.yaml per arm")
args = ap.parse_args()

PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant "
    "snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples "
    "reflect the tree and sky, creating a tranquil, meditative atmosphere."
)
arms = dict(a.split("=", 1) for a in args.arms)
extra: dict[str, dict] = {}
for e in args.env:
    arm, kv = e.split(":", 1)
    k, v = kv.split("=", 1)
    extra.setdefault(arm, {})[k] = v
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
ex = Path(args.example)


def rollout(arm_dir: str, rid: str):
    sys.path.insert(0, arm_dir)
    from mstar.client import MStarClient, VideoFrameChunk
    client = MStarClient(f"http://127.0.0.1:{args.port}")
    chunks, arrivals = [], []
    t0 = time.perf_counter()
    meta = None
    for event in client.stream(
        text=PROMPT, images=str(ex / "image.jpg"), input_modalities=("text", "image"),
        output_modalities=("video_frame",), request_id=rid, seed=42, num_frames=args.num_frames,
        camera_poses=np.load(ex / "poses.npy").tolist(),
        intrinsics=np.load(ex / "intrinsics.npy")[0].tolist(),
    ):
        assert isinstance(event, VideoFrameChunk), type(event)
        chunks.append(event.data)
        meta = event.metadata
        arrivals.append(time.perf_counter() - t0)
    h, w = meta.get("height"), meta.get("width")
    video = np.concatenate([np.frombuffer(c, dtype=np.uint8).reshape(-1, h, w, 3) for c in chunks])
    return video, arrivals


summary = []
for rnd in range(args.rounds):
    order = list(arms) if rnd % 2 == 0 else list(arms)[::-1]
    for name in order:
        d = out / f"r{rnd}_{name}"
        d.mkdir(parents=True, exist_ok=True)
        log = str(d / "server.log")
        env = {"MSTAR_PHASE_TIMING": args.period, **extra.get(name, {})}
        cfgs = dict(c.split(":", 1) for c in args.configs)
        srv = Server(arms[name], ["lingbot_world_1_3b", "--config", cfgs.get(name, "configs/lingbot_world_1_3b.yaml")],
                     args.port, log, gpus=args.gpus, env=env)
        startup = srv.start()
        try:
            rollout(arms[name], f"warm-{rnd}-{name}")
            for rep in range(args.reps):
                video, arr = rollout(arms[name], f"m-{rnd}-{name}-{rep}")
                gaps = [b - a for a, b in zip(arr, arr[1:])]
                res = {"round": rnd, "arm": name, "rep": rep, "startup_s": startup, "chunks": len(arr),
                       "frames": int(video.shape[0]), "first_chunk_s": arr[0], "total_s": arr[-1],
                       "gap_mean_s": statistics.mean(gaps), "gap_p50_s": statistics.median(gaps),
                       "fps": video.shape[0] / arr[-1], "env": env}
                summary.append(res)
                print(json.dumps(res), flush=True)
                if rep == 0:
                    np.save(d / "frames.npy", video)
        finally:
            srv.stop()
        try:
            (d / "phases.txt").write_text(phases(log))
        except Exception as exc:  # noqa: BLE001
            (d / "phases.txt").write_text(f"phase parse failed: {exc}")
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
print("AB DONE")
