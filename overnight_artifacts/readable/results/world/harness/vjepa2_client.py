"""Closed-loop client for a running vjepa2 / vjepa2_ac server.

POSTs one video N times at concurrency C to /generate, measures JCT, and saves the
float32 output of every request (requests are identical, so all should match), for
an output comparison between arms.

  python vjepa2_client.py --port 8100 --video v.webm --n 20 --c 1 --out DIR [--kwargs '{...}']
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--c", type=int, default=1)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--kwargs", default="{}")
ap.add_argument("--out", required=True)
ap.add_argument("--save", type=int, default=1, help="save outputs of the first K measured requests")
args = ap.parse_args()

url = f"http://127.0.0.1:{args.port}/generate"
video = Path(args.video).read_bytes()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def one(i: int) -> tuple[float, float | None, bytes]:
    files = [("files", (Path(args.video).name, video, "video/mp4"))]
    data = {"input_modalities": "video", "output_modalities": "video", "model_kwargs": args.kwargs}
    t0 = time.monotonic()
    first = None
    chunks = []
    with requests.post(url, data=data, files=files, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1 << 20):
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("error"):
                raise RuntimeError(msg)
            if msg.get("modality") == "video" and msg.get("data"):
                if first is None:
                    first = time.monotonic() - t0
                chunks.append(base64.b64decode(msg["data"]))
    if not chunks:
        raise RuntimeError(f"request {i}: no output chunks")
    return time.monotonic() - t0, first, b"".join(chunks)


for i in range(args.warmup * args.c):
    one(-1 - i)

lat, ttfc, failed = [], [], 0
lock = threading.Lock()
t_start = time.monotonic()


def task(i):
    global failed
    try:
        jct, first, raw = one(i)
    except Exception as exc:  # noqa: BLE001
        with lock:
            failed += 1
        print("FAIL", i, exc)
        return
    with lock:
        lat.append(jct)
        ttfc.append(first)
    if i < args.save:
        np.save(out / f"req_{i:02d}.npy", np.frombuffer(raw, dtype=np.float32))


with ThreadPoolExecutor(args.c) as ex:
    list(ex.map(task, range(args.n)))
wall = time.monotonic() - t_start
res = {
    "n": args.n, "c": args.c, "failed": failed, "wall_s": wall, "req_per_s": len(lat) / wall,
    "jct_mean_s": statistics.mean(lat) if lat else None,
    "jct_p50_s": statistics.median(lat) if lat else None,
    "jct_p90_s": sorted(lat)[int(0.9 * (len(lat) - 1))] if lat else None,
    "first_chunk_p50_s": statistics.median(ttfc) if ttfc else None,
    "kwargs": args.kwargs[:200],
}
(out / "results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
