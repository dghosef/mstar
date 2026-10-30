"""pi0.5 on DROID, mstar vs openpi (PyTorch), same box and GPU, c=1 job completion time.

mstar is served (configs/pi05_droid.yaml) and timed per HTTP request with the repo's own request path
(benchmark/run_robotics.py); openpi runs in-process via benchmark/openpi_pi05.py, so its JCT has no HTTP or image
upload in it. Both use the same lerobot/droid_100 episodes. Rounds rotate the arm order, each mstar round is a
fresh server, and a final mstar pass with MSTAR_PHASE_TIMING=100 is the profile (untimed).

  PUSH4_CORES=<openpi + client> PUSH4_MSTAR_CORES=<mstar server> run_pi05.py --gpu 3 --out ~/push4/results/newb/pi05 [--rounds 2] [--n 50] [--after-pid PID]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root: benchmark.dataset, benchmark.run_robotics
import run_gen  # noqa: E402  (Server, pin, NEWB/TREE paths)

OPENPI_PY = str(Path.home() / "push4" / "engines" / "openpi" / ".venv" / "bin" / "python")
OPENPI_CKPT = str(Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_droid_pytorch")
MEDIA = run_gen.DATA / "droid100_media"
PORT = 59300
ACTION_DIM = 32  # mstar returns the full 32-dim flow-matching chunk as float32; openpi slices it to 8 dims


def mstar_round(gpu: str, n: int, warm: int, out: Path, tag: str, profile: bool) -> dict:
    from benchmark.dataset import DROIDDataset
    from benchmark.run_robotics import _send_pi05

    m = {"mstar": "pi05", "config": "configs/pi05_droid.yaml", "gpus": "0", "visible": gpu, "comps": {}}
    env = {"MSTAR_PHASE_TIMING": "100"} if profile else {}
    reqs = DROIDDataset(local_file_dir=str(MEDIA), num_requests=n, task="pi05").get_requests()
    s = run_gen.Server(m, "mstar", env, PORT, out / f"server_{tag}.log")
    try:
        url = f"http://127.0.0.1:{PORT}/generate"
        for i in range(warm):
            _send_pi05(url, reqs[i % len(reqs)])
        lat, bad, sizes = [], 0, set()
        for req in reqs:
            raw, sec = _send_pi05(url, req)
            a = np.frombuffer(raw or b"", dtype=np.float32)
            sizes.add(a.size)
            if a.size == 0 or a.size % ACTION_DIM or not np.isfinite(a).all():
                bad += 1
                continue
            lat.append(sec * 1000.0)
    finally:
        s.stop()
    return {"jct_ms": lat, "failed": bad, "out_floats": sorted(sizes), "cores": s.cores}


def openpi_round(gpu: str, n: int, warm: int, out: Path, tag: str) -> dict:
    res = out / f"openpi_{tag}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "HF_HOME": str(Path.home() / "hf")}
    cmd = [OPENPI_PY, str(HERE.parents[0] / "openpi_pi05.py"), "--num-requests", str(n), "--num-warmup", str(warm),
           "--checkpoint", OPENPI_CKPT, "--device", "cuda:0", "--output-dir", str(res), "--local-cache", str(MEDIA)]
    with open(out / f"openpi_{tag}.log", "w") as f:
        r = subprocess.run(run_gen.pin(cmd), cwd=str(HERE.parents[1]), env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"openpi {tag} failed rc={r.returncode}; see {out / f'openpi_{tag}.log'}")
    j = json.loads((res / "results.json").read_text())
    per = j["per_request"]
    return {"jct_ms": [p["jct_ms"] for p in per if p["finite"]],
            "failed": j["failed"] + sum(1 for p in per if not p["finite"]), "cores": run_gen.CORES}


def summary(arm: str, rnd: int, d: dict) -> dict:
    x = sorted(d["jct_ms"])
    row = {"arm": arm, "round": rnd, "n": len(x), "failed": d["failed"], "out_floats": d.get("out_floats"),
           "cores": d["cores"],
           "ts": time.time()}
    if x:
        row.update({"mean_ms": statistics.fmean(x), "p50_ms": x[len(x) // 2], "p90_ms": x[int(len(x) * 0.9)],
                    "min_ms": x[0]})
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, help="physical GPU (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--after-pid", type=int, action="append", default=[],
                    help="wait for these processes (jobs ahead on the same GPU) to exit first")
    a = ap.parse_args()
    for pid in a.after_pid:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(20)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "bench.jsonl"
    arms = ["mstar", "openpi"]
    for r in range(1, a.rounds + 1):
        for arm in (arms if r % 2 else arms[::-1]):
            tag = f"r{r}"
            d = (mstar_round(a.gpu, a.n, a.warm, out, tag, False) if arm == "mstar"
                 else openpi_round(a.gpu, a.n, a.warm, out, tag))
            row = summary(arm, r, d)
            with open(rows, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[pi05 {time.strftime('%H:%M:%S')}] {json.dumps(row)}", flush=True)
    d = mstar_round(a.gpu, a.n, a.warm, out, "profile", True)
    print(f"[pi05 {time.strftime('%H:%M:%S')}] profile {json.dumps(summary('mstar_profile', 0, d))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
