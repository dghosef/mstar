"""pi0.5 on DROID, mstar vs openpi (PyTorch), same box and GPU, c=1 job completion time.

mstar is served (configs/pi05_droid.yaml) and timed per HTTP request with the repo's own request path
(benchmark/run_robotics.py); openpi runs in-process via benchmark/openpi_pi05.py, so its JCT has no HTTP or image
upload in it. Both use the same lerobot/droid_100 episodes. Rounds rotate the arm order, each mstar round is a
fresh server, and a final mstar pass with MSTAR_PHASE_TIMING=100 is the profile (untimed).

  PUSH4_CORES=<openpi + client> PUSH4_MSTAR_CORES=<mstar server> \
      run_pi05.py --gpu 3 --out ~/push4/results/newb/pi05 [--rounds 2] [--n 50] [--after-pid PID]
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
PORT = int(os.environ.get("PUSH4_PI05_PORT", "59300"))
WS_PORT = PORT + 10
OPENPI_DIR = Path.home() / "push4" / "engines" / "openpi"
ROUNDS = {"mstar": lambda a, out, tag: mstar_round(a.gpu, a.n, a.warm, out, tag, False),
          "openpi": lambda a, out, tag: openpi_round(a.gpu, a.n, a.warm, out, tag),
          "openpi_ws": lambda a, out, tag: openpi_ws_round(a.gpu, a.n, a.warm, out, tag)}
ACTION_DIM = 32  # mstar returns the full 32-dim flow-matching chunk as float32; openpi slices it to 8 dims


def mstar_round(gpu: str, n: int, warm: int, out: Path, tag: str, profile: bool, extra_env: dict | None = None) -> dict:
    from benchmark.dataset import DROIDDataset
    from benchmark.run_robotics import _send_pi05

    m = {"name": "pi05", "mstar": "pi05", "config": "configs/pi05_droid.yaml", "gpus": "0", "visible": gpu, "comps": {}}
    env = {"MSTAR_PHASE_TIMING": "100"} if profile else {}
    env.update(extra_env or {})
    reqs = DROIDDataset(local_file_dir=str(MEDIA), num_requests=n, task="pi05").get_requests()
    # numactl only binds CPUs inside the caller's affinity, so widen it to spawn the server, then narrow back to
    # the client's set so this process's HTTP client stays off the server's CPUs
    client = {int(c) for c in run_gen.CORES.split(",")}
    os.sched_setaffinity(0, client | {int(c) for c in run_gen.MSTAR_CORES.split(",")})
    try:
        s = run_gen.Server(m, "mstar", env, PORT, out / f"server_{tag}.log")
    finally:
        os.sched_setaffinity(0, client)
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
    # openpi pins an older `datasets` that can't read the mstar venv's dataset cache, so it keeps its own
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "HF_HOME": str(Path.home() / "hf"),
           "HF_DATASETS_CACHE": str(Path.home() / "hf" / "datasets_openpi")}
    cmd = [OPENPI_PY, str(HERE.parents[0] / "openpi_pi05.py"), "--num-requests", str(n), "--num-warmup", str(warm),
           "--checkpoint", OPENPI_CKPT, "--device", "cuda:0", "--output-dir", str(res), "--local-cache", str(MEDIA)]
    with open(out / f"openpi_{tag}.log", "w") as f:
        r = subprocess.run(run_gen.pin(cmd), check=False, cwd=str(HERE.parents[1]), env=env, stdout=f,
                           stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"openpi {tag} failed rc={r.returncode}; see {out / f'openpi_{tag}.log'}")
    j = json.loads((res / "results.json").read_text())
    per = j["per_request"]
    return {"jct_ms": [p["jct_ms"] for p in per if p["finite"]],
            "failed": j["failed"] + sum(1 for p in per if not p["finite"]), "cores": run_gen.CORES}


def openpi_ws_round(gpu: str, n: int, warm: int, out: Path, tag: str) -> dict:
    """openpi's own websocket policy server (scripts/serve_policy.py) and a client of it (openpi_ws_client.py)."""
    import signal
    import socket

    res = out / f"openpi_ws_{tag}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "HF_HOME": str(Path.home() / "hf"),
           "HF_DATASETS_CACHE": str(Path.home() / "hf" / "datasets_openpi")}
    srv_cmd = [OPENPI_PY, "scripts/serve_policy.py", "--port", str(WS_PORT), "policy:checkpoint",
               "--policy.config", "pi05_droid", "--policy.dir", OPENPI_CKPT]
    run_gen.wait_port_free(WS_PORT)
    with open(out / f"openpi_ws_server_{tag}.log", "w") as sf:
        srv = subprocess.Popen(run_gen.pin(srv_cmd), cwd=str(OPENPI_DIR), env=env, stdout=sf,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.monotonic()
        while True:
            if srv.poll() is not None:
                raise SystemExit(f"openpi server {tag} died at startup; see openpi_ws_server_{tag}.log")
            try:
                socket.create_connection(("127.0.0.1", WS_PORT), timeout=2).close()
                break
            except OSError:
                if time.monotonic() - t0 > 1800:
                    raise SystemExit(f"openpi server {tag} not up in 30 min") from None
                time.sleep(3)
        print(f"[pi05 {time.strftime('%H:%M:%S')}] openpi ws server up in {time.monotonic() - t0:.0f}s", flush=True)
        cmd = [OPENPI_PY, str(HERE / "openpi_ws_client.py"), "--port", str(WS_PORT), "--n", str(n),
               "--warm", str(warm), "--local-cache", str(MEDIA), "--out", str(res)]
        with open(out / f"openpi_ws_{tag}.log", "w") as f:
            r = subprocess.run(run_gen.pin(cmd), check=False, cwd=str(HERE.parents[1]), env=env, stdout=f,
                               stderr=subprocess.STDOUT)
        if r.returncode:
            raise SystemExit(f"openpi ws client {tag} failed rc={r.returncode}")
    finally:
        try:
            os.killpg(srv.pid, signal.SIGINT)
            srv.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(srv.pid, signal.SIGKILL)
    j = json.loads((res / "results.json").read_text())
    per = [p for p in j["per_request"] if p["finite"]]
    return {"jct_ms": [p["infer_ms"] for p in per], "obs_infer_ms": [p["obs_infer_ms"] for p in per],
            "failed": j["failed"] + len(j["per_request"]) - len(per), "cores": run_gen.CORES}


def summary(arm: str, rnd: int, d: dict) -> dict:
    x = sorted(d["jct_ms"])
    row = {"arm": arm, "round": rnd, "n": len(x), "failed": d["failed"], "out_floats": d.get("out_floats"),
           "cores": d["cores"],
           "ts": time.time()}
    if x:
        row.update({"mean_ms": statistics.fmean(x), "p50_ms": x[len(x) // 2], "p90_ms": x[int(len(x) * 0.9)],
                    "min_ms": x[0]})
    if d.get("obs_infer_ms"):
        y = sorted(d["obs_infer_ms"])
        row.update({"obs_infer_p50_ms": y[len(y) // 2], "obs_infer_mean_ms": statistics.fmean(y)})
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, help="physical GPU (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--arms", default="mstar,openpi",
                    help=f"from {sorted(ROUNDS)} or label=/tree (mstar from that tree); rounds rotate the order")
    ap.add_argument("--timeline", action="store_true",
                    help="instead of a bench: one mstar pass with --log-stats-file, then benchmark.request_timeline")
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
    if a.timeline:
        stats = out / "request_stats.log"
        d = mstar_round(a.gpu, a.n, a.warm, out, "timeline", False,
                        {"OVN_SERVE_EXTRA": f"--log-stats-file {stats}"})
        row = summary("mstar_timeline", 0, d)
        print(f"[pi05 {time.strftime('%H:%M:%S')}] {json.dumps(row)}", flush=True)
        r = subprocess.run([run_gen.PY, "-m", "benchmark.request_timeline.report", str(stats), "--skip", str(a.warm),
                            "--client-e2e-ms", f"{row.get('mean_ms', 0):.1f}"],
                           cwd=str(HERE.parents[1]), capture_output=True, text=True, check=False)
        (out / "timeline.txt").write_text(r.stdout + r.stderr[-3000:])
        print(r.stdout, flush=True)
        return 0
    arms = a.arms.split(",")
    for r in range(1, a.rounds + 1):
        for arm in (arms if r % 2 else arms[::-1]):
            tag = f"r{r}"
            if "=" in arm:  # label=/path/to/tree: an mstar arm served from that tree
                label, tree = arm.split("=", 1)
                d = mstar_round(a.gpu, a.n, a.warm, out, f"{label}_{tag}", False, {"OVN_TREE": tree})
                arm = label
            else:
                d = ROUNDS[arm](a, out, tag)
            row = summary(arm, r, d)
            with open(rows, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[pi05 {time.strftime('%H:%M:%S')}] {json.dumps(row)}", flush=True)
    d = mstar_round(a.gpu, a.n, a.warm, out, "profile", True)
    print(f"[pi05 {time.strftime('%H:%M:%S')}] profile {json.dumps(summary('mstar_profile', 0, d))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
