#!/usr/bin/env python3
"""One server lifetime of one arm: start under MSTAR_PHASE_TIMING, sweep, stop.

    python bench_arm.py --tree /root/work/base --label base_r1 --model qwen3_tts \
        --config configs/qwen3tts.yaml --gpu 0 --port 8100 --out /workspace/results/q06

Writes <out>/<label>/{server.log,phases.txt,c<C>.json,c<C>_wav/,env.txt}.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VENV = "/root/work/mstar/.venv"
HARNESS = Path(__file__).resolve().parent


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concs", default="1,16,32,64")
    ap.add_argument("--n1", type=int, default=30, help="sentences at c=1")
    ap.add_argument("--sentences", default=str(HARNESS / "sentences_260.txt"))
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--language", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-audio", default="1,16")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL for the server")
    ap.add_argument("--period", type=int, default=100)
    ap.add_argument("--health-timeout", type=int, default=2400)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--extra-file", action="append", default=[], help="KEY=PATH: field value read from a file")
    ap.add_argument("--greedy", default="", help="comma list of KEY=JSON for an untimed greedy c=1 pass")
    ap.add_argument("--greedy-n", type=int, default=20)
    ap.add_argument("--pyspy-cell", type=int, default=0, help="py-spy the GPU0 worker(s) during this concurrency cell")
    ap.add_argument("--pyspy-secs", type=int, default=20)
    args = ap.parse_args()

    dest = Path(args.out) / args.label
    dest.mkdir(parents=True, exist_ok=True)
    if healthy(args.port):
        print(f"port {args.port} already serving; refusing", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env.update({
        "PATH": f"{VENV}/bin:" + env.get("PATH", ""),
        "VIRTUAL_ENV": VENV,
        "PYTHONPATH": ":".join(x for x in (args.tree, os.environ.get("GUEST_PYLIB", "")) if x),
        "HF_HOME": "/root/work/hf",
        "MSTAR_SHM_ARENA": "1",
        "CUDA_HOME": "/usr/local/cuda-12.9",
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("GUEST_INDUCTOR", "/root/work/inductor_cache"),
        "TORCHINDUCTOR_COMPILE_THREADS": "8",
        "MAX_JOBS": "16",
    })
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    py = f"{VENV}/bin/python"
    imported = subprocess.run([py, "-c", "import mstar; print(mstar.__file__)"], cwd=args.tree,
                              env=env, capture_output=True, text=True).stdout.strip()
    serve = (f"{py} -m mstar.cli.main serve {args.model} --config {args.config} "
             f"--gpus {args.gpu} --port {args.port}")
    (dest / "env.txt").write_text(json.dumps({
        "tree": args.tree, "imported": imported, "serve": serve, "env_extra": args.env,
        "started": time.strftime("%F %T"), "load": os.getloadavg(),
    }, indent=1))
    print(f"[{args.label}] imported {imported}", flush=True)
    wrapper = subprocess.Popen(
        [py, "-m", "benchmark.worker_phases.server", "--command", serve,
         "--period", str(args.period), "--server-log", str(dest / "server.log"),
         "--out", str(dest / "phases.txt"), "--every", "1000000"],
        cwd=args.tree, env=env, start_new_session=True,
        stdout=open(dest / "wrapper.log", "w"), stderr=subprocess.STDOUT,
    )
    t0 = time.time()
    rc = 0
    try:
        while not healthy(args.port):
            if wrapper.poll() is not None:
                print(f"[{args.label}] server died; see {dest/'server.log'}", flush=True)
                return 2
            if time.time() - t0 > args.health_timeout:
                print(f"[{args.label}] health timeout", flush=True)
                return 3
            time.sleep(5)
        print(f"[{args.label}] ready after {time.time()-t0:.0f}s load_start={os.getloadavg()[0]:.0f}", flush=True)
        client = ([py, str(HARNESS / "speech_bench_ext.py")] if args.extra_file
                  else [py, "-m", "benchmark.tts_speech_bench"])
        base = [*client, "--engine", "mstar",
                "--url", f"http://127.0.0.1:{args.port}", "--model", args.model,
                "--sentences", args.sentences, "--repeats", "1",
                "--sample-rate", str(args.sample_rate), "--seed", str(args.seed)]
        if args.voice:
            base += ["--voice", args.voice]
        if args.language:
            base += ["--language", args.language]
        for e in args.extra:
            base += ["--extra", e]
        for e in args.extra_file:
            base += ["--extra-file", e]
        # Warmup at the highest concurrency (compiles, captures and the
        # first-of-shape costs land here, not in a measured cell).
        top = max(int(c) for c in args.concs.split(","))
        subprocess.run(base + ["--concurrency", str(min(top, 32)), "--warmup", "0",
                               "--num-sentences", str(2 * min(top, 32)),
                               "--out", str(dest / "warmup.json")],
                       cwd=args.tree, env=env, stdout=open(dest / "warmup.log", "w"),
                       stderr=subprocess.STDOUT)
        saves = {int(c) for c in args.save_audio.split(",") if c}
        for c in (int(x) for x in args.concs.split(",")):
            n = args.n1 if c == 1 else (130 if c <= 16 else 260)
            cmd = base + ["--concurrency", str(c), "--warmup", str(min(c, 8)),
                          "--num-sentences", str(n), "--out", str(dest / f"c{c}.json"),
                          "--label", f"{args.label} c{c}"]
            if c in saves:
                cmd += ["--save-audio-dir", str(dest / f"c{c}_wav")]
            spy = None
            if c == args.pyspy_cell:
                spy = subprocess.Popen(["bash", "-c", (
                    "sleep 8; for pid in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do "
                    f"timeout {args.pyspy_secs + 30} {VENV}/bin/py-spy record --nonblocking -r 50 -d {args.pyspy_secs} -f raw "
                    f"-o {dest}/pyspy_c{c}_$pid.txt -p $pid > {dest}/pyspy_$pid.log 2>&1 & done; wait")])
            p = subprocess.run(cmd, cwd=args.tree, env=env, capture_output=True, text=True)
            if spy is not None:
                spy.wait()
            (dest / f"c{c}.log").write_text(p.stdout + p.stderr)
            print(f"[{args.label}] {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]}",
                  flush=True)
        if args.greedy:
            cmd = base + ["--concurrency", "1", "--warmup", "0", "--num-sentences", str(args.greedy_n),
                          "--out", str(dest / "greedy.json"), "--save-audio-dir", str(dest / "greedy_wav")]
            for kv in args.greedy.split(","):
                cmd += ["--extra", kv]
            p = subprocess.run(cmd, cwd=args.tree, env=env, capture_output=True, text=True)
            (dest / "greedy.log").write_text(p.stdout + p.stderr)
            print(f"[{args.label}] greedy done", flush=True)
    finally:
        try:
            os.killpg(wrapper.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            wrapper.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(wrapper.pid, signal.SIGKILL)
            rc = 4
        for _ in range(60):
            if not healthy(args.port):
                break
            time.sleep(2)
        print(f"[{args.label}] stopped load_end={os.getloadavg()[0]:.0f}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
