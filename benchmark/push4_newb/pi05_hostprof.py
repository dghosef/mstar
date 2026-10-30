"""Where pi0.5's ~73 ms of API-server time per request goes (recv -> preprocess done in the request timeline).

Starts a pi0.5 mstar server under `py-spy record` (spawn mode: attaching is denied in the push4c container), sends
warmup + N requests at c=1 with the bench's request path, stops it, and summarizes the samples of the API-server
threads that run request preprocessing (any stack through mstar/api_server/data_worker.py), by the deepest mstar
frame and by leaf.

  PUSH4_MSTAR_CORES=106,108 PUSH4_CORES=110,...,206 pi05_hostprof.py --gpu 1 --out DIR [--n 50]
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import run_gen  # noqa: E402
import run_pi05  # noqa: E402

PYSPY = str(Path(run_gen.PY).parent / "py-spy")
PORT = 59320
FRAME = re.compile(r"^(.*) \((.*?):(\d+)\)$")


def summarize(raw: Path, top: int = 25) -> str:
    deepest_mstar, leaf, thread_total = Counter(), Counter(), Counter()
    total = 0
    for line in raw.read_text().splitlines():
        stack, _, n = line.rpartition(" ")
        if "data_worker.py" not in stack:
            continue
        n = int(n)
        total += n
        frames = stack.split(";")
        thread = next((f for f in frames if f.startswith("thread ")), "?")
        thread_total[thread] += n
        parsed = [m.groups() for f in frames if (m := FRAME.match(f))]
        mine = [f"{fn} ({Path(path).name}:{ln})" for fn, path, ln in parsed if "/mstar/" in path]
        if mine:
            deepest_mstar[mine[-1]] += n
        if parsed:
            fn, path, ln = parsed[-1]
            leaf[f"{fn} ({Path(path).name}:{ln})"] += n
    out = [f"samples in stacks through data_worker.py: {total}", "", "by thread:"]
    out += [f"  {n:7d}  {t}" for t, n in thread_total.most_common(8)]
    out += ["", f"by deepest mstar frame (top {top}):"]
    out += [f"  {n:7d}  {100 * n / max(total, 1):5.1f}%  {f}" for f, n in deepest_mstar.most_common(top)]
    out += ["", f"by leaf frame (top {top}):"]
    out += [f"  {n:7d}  {100 * n / max(total, 1):5.1f}%  {f}" for f, n in leaf.most_common(top)]
    return "\n".join(out) + "\n"


def main() -> int:
    from benchmark.dataset import DROIDDataset
    from benchmark.run_robotics import _send_pi05

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--rate", type=int, default=250)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "pyspy_raw.txt"
    reqs = DROIDDataset(local_file_dir=str(run_pi05.MEDIA), num_requests=a.n, task="pi05").get_requests()
    m = {"name": "pi05", "mstar": "pi05", "config": "configs/pi05_droid.yaml", "gpus": "0", "visible": a.gpu,
         "comps": {}}
    wrap = f"{PYSPY} record --rate {a.rate} --nonblocking --threads --format raw --duration 1800 -o {raw} --"
    import os
    client = {int(c) for c in run_gen.CORES.split(",")}
    os.sched_setaffinity(0, client | {int(c) for c in run_gen.MSTAR_CORES.split(",")})
    try:
        s = run_gen.Server(m, "mstar", {"OVN_WRAP": wrap}, PORT, out / "server.log")
    finally:
        os.sched_setaffinity(0, client)
    lat = []
    try:
        url = f"http://127.0.0.1:{PORT}/generate"
        for i in range(a.warm):
            _send_pi05(url, reqs[i % len(reqs)])
        t0 = time.time()
        for req in reqs:
            raw_bytes, sec = _send_pi05(url, req)
            if raw_bytes:
                lat.append(sec * 1000)
        print(f"{len(lat)}/{len(reqs)} ok, p50 {statistics.median(lat):.1f} ms, load window {time.time() - t0:.1f} s",
              flush=True)
    finally:
        s.stop()
    for _ in range(60):  # py-spy writes its file after the child exits
        if raw.exists() and raw.stat().st_size:
            break
        time.sleep(2)
    text = summarize(raw)
    (out / "summary.txt").write_text(f"p50 JCT under py-spy {statistics.median(lat):.1f} ms, n={len(lat)}\n\n" + text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
