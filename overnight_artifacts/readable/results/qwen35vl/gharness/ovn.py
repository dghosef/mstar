"""Overnight qwen35vl bench driver: start a server from a given tree, run client
commands against it, stop it, and keep everything per run directory.

    python ovn.py run --tree /root/work/base --arm base --serve qwen3_5_0.8b \
        --config configs/qwen3_5_0.8b.yaml --out /workspace/results/x \
        --phase --client "i2t_c1::python -m benchmark.runner ... --url {url}"

Each --client is "label::command"; {url} and {out} are substituted. Client
commands run with cwd=--tree and PYTHONPATH=--tree, so the arm's own benchmark
code is used. The server log is kept; with --phase, MSTAR_PHASE_TIMING=100 is
set and the phase tables are rendered into phases.txt after shutdown.
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


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def gpu_mem_used(gpu: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.split()
    return sum(int(x) for x in out) if out else -1


def env_for(tree: str, extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{VENV}/bin:" + env.get("PATH", "")
    env["VIRTUAL_ENV"] = VENV
    env["PYTHONPATH"] = os.environ.get("QVL_PREPEND", "") + tree
    env["HF_HOME"] = "/root/work/hf"
    env["MSTAR_SHM_ARENA"] = "1"
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "8")
    env.setdefault("MAX_JOBS", "16")
    env.update(extra)
    return env


def reap(pids=()) -> None:
    """Kill the given pids (our own server tree) — never pattern-kill on a shared box."""
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass


def stop(proc: subprocess.Popen, gpu: int) -> None:
    tree = descendants(proc.pid)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=30)
    time.sleep(3)
    reap([p for p in tree if os.path.exists(f"/proc/{p}")])
    for _ in range(30):
        if gpu_mem_used(gpu) < 2000:
            break
        time.sleep(2)


def descendants(pid: int) -> list[str]:
    out, todo = [], [str(pid)]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += subprocess.run(["pgrep", "-P", p], capture_output=True, text=True).stdout.split()
    return out


def spy(cdir: Path, server_pid: int, delay: float = 5, dur: int = 10) -> None:
    """py-spy every python process of the server tree for ``dur`` s, after ``delay`` s of load."""
    time.sleep(delay)
    ps = []
    for pid in descendants(server_pid):
        try:
            cmdline = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")[:300]
        except OSError:
            continue
        if "python" not in cmdline or "resource_tracker" in cmdline or "compile_worker" in cmdline:
            continue
        (cdir / f"pyspy_{pid}.cmd").write_text(cmdline)
        ps.append(subprocess.Popen([f"{VENV}/bin/py-spy", "record", "-p", pid, "-d", str(dur), "-r", "100",
                                    "--format", "raw", "-o", str(cdir / f"pyspy_{pid}.txt"), "--threads", "--nonblocking"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in ps:
        p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--tree", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--serve", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", action="store_true")
    ap.add_argument("--phase-period", default="100")
    ap.add_argument("--pyspy", default=None, help="label of the client during which to py-spy the api server + worker")
    ap.add_argument("--env", action="append", default=[], help="K=V for the server")
    ap.add_argument("--serve-arg", action="append", default=[])
    ap.add_argument("--client", action="append", default=[])
    ap.add_argument("--health-timeout", type=int, default=2400)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if healthy(args.port):
        print(f"port {args.port} already serving; refusing", flush=True)
        return 1
    extra = dict(kv.split("=", 1) for kv in args.env)
    if args.phase:
        extra["MSTAR_PHASE_TIMING"] = args.phase_period
    env = env_for(args.tree, extra)
    cmd = [f"{VENV}/bin/python", "-m", "mstar.cli.main", "serve", args.serve,
           "--gpus", str(args.gpu), "--port", str(args.port)]
    if args.config:
        cmd += ["--config", args.config]
    cmd += args.serve_arg
    meta = {
        "arm": args.arm, "tree": args.tree, "cmd": cmd, "env_extra": extra,
        "start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loadavg_start": open("/proc/loadavg").read().strip(),
        "git": subprocess.run(["git", "-C", args.tree, "log", "--oneline", "-1"],
                              capture_output=True, text=True).stdout.strip(),
    }
    imp = subprocess.run([f"{VENV}/bin/python", "-c", "import mstar; print(mstar.__file__)"],
                         env=env, cwd=args.tree, capture_output=True, text=True)
    meta["mstar_file"] = imp.stdout.strip()
    print(json.dumps(meta), flush=True)
    log = open(out / "server.log", "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=args.tree, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    rc = 0
    try:
        while not healthy(args.port):
            if proc.poll() is not None:
                print(f"server died rc={proc.returncode}", flush=True)
                subprocess.run(["tail", "-50", str(out / "server.log")])
                return 2
            if time.time() - t0 > args.health_timeout:
                print("health timeout", flush=True)
                return 3
            time.sleep(3)
        meta["startup_s"] = round(time.time() - t0, 1)
        print(f"READY after {meta['startup_s']}s", flush=True)
        url = f"http://127.0.0.1:{args.port}"
        for spec in args.client:
            label, c = spec.split("::", 1)
            cdir = out / label
            cdir.mkdir(exist_ok=True)
            c = c.replace("{url}", url).replace("{out}", str(cdir))
            cenv = env_for(args.tree, {})
            print(f"--- client {label}: {c}", flush=True)
            la = open("/proc/loadavg").read().strip()
            tc = time.time()
            with open(cdir / "client.log", "w") as f:
                f.write(f"# loadavg {la}\n# {c}\n")
                f.flush()
                cp = subprocess.Popen(shlex.split(c), cwd=args.tree, env=cenv, stdout=f,
                                      stderr=subprocess.STDOUT)
                if args.pyspy == label:
                    spy(cdir, proc.pid)
                cp.wait()
                r = cp
            print(f"    rc={r.returncode} {time.time() - tc:.1f}s", flush=True)
            if r.returncode != 0:
                rc = 4
            if proc.poll() is not None:
                print("server died during client", flush=True)
                return 5
    finally:
        stop(proc, args.gpu)
        log.close()
        meta["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["loadavg_end"] = open("/proc/loadavg").read().strip()
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        if args.phase:
            try:
                sys.path.insert(0, args.tree)
                from benchmark.worker_phases.phases import parse_log, render, segment
                txt = (out / "server.log").read_text(errors="replace")
                (out / "phases.txt").write_text(render(segment(parse_log(txt))))
            except Exception as e:  # noqa: BLE001
                print("phase render failed", e, flush=True)
    print("RUN DONE", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
