"""nsys-profile a running Waypoint server's steady-state rollout, then summarize kernels.

  python wp_nsys.py --arm /root/guest-world/base --tag base --variant 720p --streams 1 --steps 48
Server runs under `nsys launch` (cuda-graph node trace); collection starts after a warmup
stream, covers one measured wave of `--streams` concurrent rollouts, and stops.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--variant", default="720p")
ap.add_argument("--streams", type=int, default=1)
ap.add_argument("--steps", type=int, default=48)
ap.add_argument("--port", type=int, default=9102)
ap.add_argument("--seed-image", default="/root/work/media/seed.jpg")
ap.add_argument("--gpus", default="0")
ap.add_argument("--nvtx", action="store_true")
args = ap.parse_args()

sys.path.insert(0, f"{args.arm}/test/waypoint")
sys.path.insert(0, args.arm)
import serve_rollout as rollout  # noqa: E402

from mstar.client import MStarClient  # noqa: E402

out = Path(f"/workspace/results/world/waypoint_nsys/{args.tag}")
out.mkdir(parents=True, exist_ok=True)
variant = rollout.VARIANTS[args.variant]
work = Path(tempfile.mkdtemp(prefix="wpnsys-"))
cfg_path = Path(args.arm) / ("configs/waypoint.yaml" if args.variant == "720p" else "configs/waypoint_360p.yaml")
cfg = rollout._run_config(cfg_path, variant, None, None, work / "run.yaml",
                          worlds=max(args.streams, 1), batch=max(args.streams, 1))
seed = rollout._seed_png(Path(args.seed_image), variant, work / "seed.png")
session = f"wp{os.getpid()}"
cmd = ["nsys", "launch", f"--session={session}", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
       sys.executable, f"{args.arm}/mstar/api_server/entrypoint.py", "--config", str(cfg), "--port",
       str(args.port), "--host", "127.0.0.1", "--socket-path-prefix", str(work / "sock"),
       "--upload-dir", str(work / "up"), "--tensor-comm-protocol", "SHM", *(["--enable-nvtx"] if args.nvtx else [])]
env = {**os.environ, "PYTHONPATH": args.arm, "CUDA_VISIBLE_DEVICES": args.gpus, "MSTAR_SHM_ARENA": "1",
       "HF_HOME": "/root/work/hf"}
log = open(out / "server.log", "w")
proc = subprocess.Popen(cmd, cwd=args.arm, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
client = MStarClient(f"http://127.0.0.1:{args.port}")
try:
    rollout._wait_for_health(client, proc, 1800)
    rollout._rollout(client, seed, 8, "warm", 7)
    time.sleep(2)
    subprocess.run(["nsys", "start", f"--session={session}", "-o", str(out / "trace"), "-f", "true"],
                   check=True)
    t0 = time.time()
    threads = [threading.Thread(target=rollout._rollout, args=(client, seed, args.steps, f"m{i}", 100 + i))
               for i in range(args.streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    subprocess.run(["nsys", "stop", f"--session={session}"], check=True)
finally:
    import signal
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=90)
    except Exception:  # noqa: BLE001
        os.killpg(proc.pid, signal.SIGKILL)
sys.path.insert(0, str(Path(__file__).parent))
from nsys_summary import summarize  # noqa: E402

print(summarize(out / "trace.nsys-rep", f"tag={args.tag} streams={args.streams} steps={args.steps} wall={wall:.3f}s",
                args.steps * args.streams))
