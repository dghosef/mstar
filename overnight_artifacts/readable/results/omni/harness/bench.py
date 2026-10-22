#!/usr/bin/env python3
"""Qwen3-Omni bench driver for the overnight push (runs on the box).

One invocation = one server process (one arm) + a list of client workloads.

  python bench.py --tree /root/work/base --arm base --tag r1 \
      --config configs/qwen3omni_2gpu.yaml --workloads t2s_c1,t2s_c8 [--phase-timing 100]

Results land in /workspace/results/<tag>/<arm>/<workload>/ (bench.log, results.json,
req_*.txt/.wav) plus server.log and arm.json (which mstar was imported, load avg).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

VENV = "/root/work/mstar/.venv"

# name -> runner args. N >= 5x concurrency where affordable.
WORKLOADS = {
    "t2s_c1": dict(rt="text_to_speech", c=1, n=6, w=1),
    "t2s_c8": dict(rt="text_to_speech", c=8, n=32, w=1),
    "t2s_c16": dict(rt="text_to_speech", c=16, n=64, w=1),
    "t2s_c32": dict(rt="text_to_speech", c=32, n=96, w=1),
    "t2t_c1": dict(rt="text_to_text", c=1, n=8, w=1),
    "t2t_c16": dict(rt="text_to_text", c=16, n=64, w=1),
    "t2t_c32": dict(rt="text_to_text", c=32, n=96, w=1),
    "a2s_c1": dict(rt="audio_to_speech", c=1, n=12, w=1, ds="libri"),
    "a2s_c8": dict(rt="audio_to_speech", c=8, n=32, w=1, ds="libri"),
    "a2s_c16": dict(rt="audio_to_speech", c=16, n=64, w=1, ds="libri"),
    "a2t_c16": dict(rt="audio_to_text", c=16, n=64, w=1, ds="libri"),
    "i2t_c1": dict(rt="image_to_text", c=1, n=8, w=1, ds="food101"),
    "i2t_c16": dict(rt="image_to_text", c=16, n=64, w=1, ds="food101"),
    "i2s_c8": dict(rt="image_to_speech", c=8, n=32, w=1, ds="food101"),
}


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def gpu_mem() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def env_for(tree: str, extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{VENV}/bin:" + env.get("PATH", "")
    env["VIRTUAL_ENV"] = VENV
    env["HF_HOME"] = "/root/work/hf"
    env["MSTAR_SHM_ARENA"] = "1"
    env["PYTHONPATH"] = tree + os.environ.get("BENCH_EXTRA_PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    env["MAX_JOBS"] = "16"
    env.update(extra)
    return env


def stop(proc: subprocess.Popen, port: int) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        for _ in range(90):
            if proc.poll() is not None:
                break
            time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # Reap orphans of this server (spawned workers keep the session id).
    subprocess.run(["pkill", "-9", "-s", str(proc.pid)], capture_output=True)
    for _ in range(60):
        if max(gpu_mem()) < 2000:
            break
        time.sleep(2)
    print(f"[bench] stopped, gpu mem {gpu_mem()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default="configs/qwen3omni_2gpu.yaml")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--workloads", required=True)
    ap.add_argument("--phase-timing", type=int, default=0)
    ap.add_argument("--env", action="append", default=[], help="K=V for the server")
    ap.add_argument("--startup-timeout", type=int, default=3000)
    ap.add_argument("--extra-client", default="", help="extra runner args")
    args = ap.parse_args()

    root = f"/workspace/results/{args.tag}/{args.arm}"
    os.makedirs(root, exist_ok=True)
    extra = dict(kv.split("=", 1) for kv in args.env)
    if args.phase_timing:
        extra["MSTAR_PHASE_TIMING"] = str(args.phase_timing)
    env = env_for(args.tree, extra)

    if healthy(args.port):
        sys.exit(f"port {args.port} already serving")
    which = subprocess.run([f"{VENV}/bin/python", "-c", "import mstar;print(mstar.__file__)"],
                           cwd=args.tree, env=env, capture_output=True, text=True).stdout.strip()
    meta = {"tree": args.tree, "mstar_file": which, "config": args.config, "env": extra,
            "load_start": open("/proc/loadavg").read().strip(), "t_start": time.time()}
    print(f"[bench] arm={args.arm} mstar={which}", flush=True)
    assert which.startswith(args.tree), which

    slog = open(f"{root}/server.log", "w")
    cmd = [f"{VENV}/bin/python", "-m", "mstar.cli.main", "serve", "qwen3_omni",
           "--config", args.config, "--gpus", args.gpus, "--port", str(args.port)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=args.tree, env=env, stdout=slog, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        while not healthy(args.port):
            if proc.poll() is not None:
                raise RuntimeError(f"server died rc={proc.returncode}; see {root}/server.log")
            if time.time() - t0 > args.startup_timeout:
                raise RuntimeError("startup timeout")
            time.sleep(5)
        meta["startup_s"] = time.time() - t0
        print(f"[bench] ready after {meta['startup_s']:.0f}s", flush=True)
        meta["workloads"] = {}
        for wl in args.workloads.split(","):
            spec = WORKLOADS[wl]
            out = f"{root}/{wl}"
            os.makedirs(out, exist_ok=True)
            ccmd = [f"{VENV}/bin/python", "-m", "benchmark.runner",
                    "--url", f"http://127.0.0.1:{args.port}", "--model", "qwen3omni",
                    "--profiling-type", "closed_loop", "--request-type", spec["rt"],
                    "--num-requests", str(spec["n"]), "--max-concurrency", str(spec["c"]),
                    "--num-warmup", str(spec["w"]), "--inference-system", "ours",
                    "--dataset", spec.get("ds", "text"), "--output-dir", out,
                    "--local-cache", "/root/work/bench-cache"]
            if args.extra_client:
                ccmd += args.extra_client.split()
            tw = time.time()
            with open(f"{out}/bench.log", "w") as f:
                r = subprocess.run(ccmd, cwd=args.tree, env=env_for(args.tree, {}), stdout=f,
                                   stderr=subprocess.STDOUT, timeout=5400)
            meta["workloads"][wl] = {"rc": r.returncode, "wall_s": time.time() - tw,
                                     "load": open("/proc/loadavg").read().strip()}
            print(f"[bench] {wl} rc={r.returncode} {time.time()-tw:.0f}s", flush=True)
            if proc.poll() is not None:
                raise RuntimeError("server died during client")
    finally:
        meta["t_end"] = time.time()
        json.dump(meta, open(f"{root}/arm.json", "w"), indent=1)
        stop(proc, args.port)
        slog.close()
    print("[bench] ARM DONE", flush=True)


if __name__ == "__main__":
    main()
