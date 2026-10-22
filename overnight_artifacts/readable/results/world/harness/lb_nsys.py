"""nsys-profile one steady LingBot rollout on a running server (after a warmup rollout).

  python lb_nsys.py --arm /root/work/base --tag base --gpus 0
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nsys_summary import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--gpus", default="0")
ap.add_argument("--port", type=int, default=9130)
ap.add_argument("--num-frames", type=int, default=93)
ap.add_argument("--example", default="/root/work/lingbot-world/examples/03")
ap.add_argument("--out-root", default="/workspace/results/world/lingbot_nsys")
ap.add_argument("--env", nargs="*", default=[])
args = ap.parse_args()
sys.path.insert(0, args.arm)
import numpy as np  # noqa: E402

from mstar.client import MStarClient  # noqa: E402

PROMPT = ("A serene lakeside scene with a lone tree standing in calm water, surrounded by distant "
          "snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples "
          "reflect the tree and sky, creating a tranquil, meditative atmosphere.")
out = Path(args.out_root) / args.tag
out.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="lbnsys-"))
session = f"lb{os.getpid()}"
py = sys.executable
cmd = ["nsys", "launch", f"--session={session}", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
       py, f"{args.arm}/mstar/api_server/entrypoint.py", "--config", f"{args.arm}/configs/lingbot_world_1_3b.yaml",
       "--port", str(args.port), "--host", "127.0.0.1", "--socket-path-prefix", str(work / "sock"),
       "--upload-dir", str(work / "up"), "--tensor-comm-protocol", "SHM"]
env = {**os.environ, "PYTHONPATH": args.arm, "CUDA_VISIBLE_DEVICES": args.gpus, "MSTAR_SHM_ARENA": "1",
       "HF_HOME": "/root/work/hf", **dict(e.split("=", 1) for e in args.env)}
log = open(out / "server.log", "w")
proc = subprocess.Popen(cmd, cwd=args.arm, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
client = MStarClient(f"http://127.0.0.1:{args.port}")
ex = Path(args.example)


def rollout(rid):
    n = 0
    for ev in client.stream(text=PROMPT, images=str(ex / "image.jpg"), input_modalities=("text", "image"),
                            output_modalities=("video_frame",), request_id=rid, seed=42,
                            num_frames=args.num_frames, camera_poses=np.load(ex / "poses.npy").tolist(),
                            intrinsics=np.load(ex / "intrinsics.npy")[0].tolist()):
        n += 1
    return n


try:
    t0 = time.time()
    while not client.health():
        if proc.poll() is not None or time.time() - t0 > 1800:
            raise SystemExit("server failed")
        time.sleep(2)
    rollout("warm")
    subprocess.run(["nsys", "start", f"--session={session}", "-o", str(out / "trace"), "-f", "true"], check=True)
    t = time.time()
    chunks = rollout("meas")
    wall = time.time() - t
    subprocess.run(["nsys", "stop", f"--session={session}"], check=True)
finally:
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=90)
    except Exception:  # noqa: BLE001
        os.killpg(proc.pid, signal.SIGKILL)
print(summarize(out / "trace.nsys-rep", f"tag={args.tag} wall={wall:.3f}s chunks={chunks}", chunks))
