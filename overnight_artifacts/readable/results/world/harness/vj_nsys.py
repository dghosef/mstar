"""nsys-profile V-JEPA2(-AC) requests on a running server (after warmup requests).

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
ap.add_argument("--model", default="vjepa2_ac")
ap.add_argument("--video", default="/root/work/media/droid_like_8f.mp4")
ap.add_argument("--kwargs", default="/root/work/wharness/ac_kwargs_h4.json")
ap.add_argument("--iters", type=int, default=6)
ap.add_argument("--nvtx", action="store_true")
ap.add_argument("--out-root", default="/workspace/results/world/vjepa2_nsys")
ap.add_argument("--env", nargs="*", default=[])
args = ap.parse_args()
sys.path.insert(0, args.arm)

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
       py, f"{args.arm}/mstar/api_server/entrypoint.py", "--config", f"{args.arm}/configs/{args.model}.yaml",
       "--port", str(args.port), "--host", "127.0.0.1", "--socket-path-prefix", str(work / "sock"),
       "--upload-dir", str(work / "up"), "--tensor-comm-protocol", "SHM", *(["--enable-nvtx"] if args.nvtx else [])]
env = {**os.environ, "PYTHONPATH": args.arm, "CUDA_VISIBLE_DEVICES": args.gpus, "MSTAR_SHM_ARENA": "1",
       "HF_HOME": "/root/work/hf", **dict(e.split("=", 1) for e in args.env)}
log = open(out / "server.log", "w")
proc = subprocess.Popen(cmd, cwd=args.arm, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
client = MStarClient(f"http://127.0.0.1:{args.port}")
import requests  # noqa: E402

vid = Path(args.video).read_bytes()
kw = Path(args.kwargs).read_text() if args.kwargs.endswith(".json") else args.kwargs


def rollout(rid):
    n = 0
    for i in range(args.iters):
        files = [("files", ("v.mp4", vid, "video/mp4"))]
        data = {"input_modalities": "video", "output_modalities": "video", "model_kwargs": kw}
        with requests.post(f"http://127.0.0.1:{args.port}/generate", data=data, files=files, stream=True) as r:
            for _ in r.iter_lines(chunk_size=1 << 20):
                pass
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
