#!/usr/bin/env python3
"""nsys capture of a steady-state window of a Qwen3-Omni server (runs on the box).

  python nsys_run.py --tree /root/work/base --out /workspace/results/nsys/base_t2s_c8 \
      --workload t2s_c8 [--seconds 8]

The server runs under `nsys launch` (CUDA trace with graph-node granularity); a warmup
client runs first, then a second client starts and `nsys start/stop` brackets a window
of it. Exports sqlite and prints the per-process kernel table.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench import VENV, WORKLOADS, env_for, healthy, stop  # noqa: E402


def client(tree, port, wl, out, env):
    spec = WORKLOADS[wl]
    cmd = [f"{VENV}/bin/python", "-m", "benchmark.runner", "--url", f"http://127.0.0.1:{port}",
           "--model", "qwen3omni", "--profiling-type", "closed_loop", "--request-type", spec["rt"],
           "--num-requests", str(spec["n"]), "--max-concurrency", str(spec["c"]),
           "--num-warmup", "0", "--inference-system", "ours", "--dataset", spec.get("ds", "text"),
           "--output-dir", out, "--local-cache", "/root/work/bench-cache"]
    return subprocess.Popen(cmd, cwd=tree, env=env, stdout=open(f"{out}/client.log", "w"),
                            stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workload", default="t2s_c8")
    ap.add_argument("--config", default="configs/qwen3omni_2gpu.yaml")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--delay", type=float, default=12)
    ap.add_argument("--env", action="append", default=[])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env = env_for(args.tree, dict(kv.split("=", 1) for kv in args.env))
    sess = f"omni{os.getpid()}"
    cmd = ["nsys", "launch", f"--session-new={sess}", "--trace=cuda,nvtx",
           "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
           f"{VENV}/bin/python", "-m", "mstar.cli.main", "serve", "qwen3_omni",
           "--config", args.config, "--gpus", "0,1", "--port", str(args.port)]
    proc = subprocess.Popen(cmd, cwd=args.tree, env=env, stdout=open(f"{args.out}/server.log", "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while not healthy(args.port):
            if proc.poll() is not None:
                raise RuntimeError("server died")
            if time.time() - t0 > 3000:
                raise RuntimeError("timeout")
            time.sleep(5)
        print(f"ready {time.time()-t0:.0f}s", flush=True)
        os.makedirs(f"{args.out}/warm", exist_ok=True)
        client(args.tree, args.port, args.workload, f"{args.out}/warm", env).wait()
        os.makedirs(f"{args.out}/cap", exist_ok=True)
        c = client(args.tree, args.port, args.workload, f"{args.out}/cap", env)
        time.sleep(args.delay)
        subprocess.run(["nsys", "start", f"--session={sess}", "-o", f"{args.out}/report", "--force-overwrite=true"], check=True)
        time.sleep(args.seconds)
        subprocess.run(["nsys", "stop", f"--session={sess}"], check=True)
        print("captured", flush=True)
        c.wait()
    finally:
        stop(proc, args.port)
        subprocess.run(["nsys", "shutdown", f"--session={sess}"], capture_output=True)
    subprocess.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true",
                    "-o", f"{args.out}/report.sqlite", f"{args.out}/report.nsys-rep"], check=True)
    subprocess.run([f"{VENV}/bin/python", os.path.join(os.path.dirname(__file__), "nsys_kernels.py"),
                    f"{args.out}/report.sqlite", "45"], stdout=open(f"{args.out}/kernels.txt", "w"))
    print("NSYS DONE", flush=True)


if __name__ == "__main__":
    main()
