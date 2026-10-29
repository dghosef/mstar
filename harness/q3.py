"""Server lifecycle + cells for the Qwen3-TTS push (runs on the box).

  q3.py serve <arm> [--gpu 0] [--port 8100] [--tag T] [--env K=V ...]   start, wait healthy, smoke; prints the log path
  q3.py stop                                                            stop the server this harness started
  q3.py cell <arm> <kind> <out-subdir> -- <slo_bench args>             serve -> slo_bench -> stop (one cell, one server)

Arms are checkouts or competitor venvs; every start goes through here so ports, GPUs, logs and
provenance are recorded the same way. A start refuses a GPU holding > 2 GB or a busy port.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(os.environ.get("Q3_BASE", "/workspace"))  # local NVMe on a borrowed box whose /workspace is not ours
W = BASE / "q3"
CACHE = BASE / "cache"
ENG = W / "engines"
RES = W / "results"
HARN = W / "harness"
CLIENT_PY = W / "clientenv/bin/python"
def state_file(gpu: int) -> Path:  # one server per GPU; GPUs run independent queues
    return W / f"server_state_gpu{gpu}.json"


STATE = state_file(0)
HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# every built mstar checkout under W is an arm (main, stack, and heads shipped by push_head.py)
MSTAR_TREES = {d.name: d for d in sorted(W.iterdir()) if (d / ".venv/bin/mstar").exists()} if W.exists() else {}


def arm_cmd(arm: str, gpu: int, port: int, extra: list[str]) -> tuple[list[str], dict, Path]:
    env = dict(os.environ, HF_HOME=str(BASE / "hf"), PYTHONUNBUFFERED="1")
    base = arm.split("+")[0]
    if base in MSTAR_TREES:
        tr = MSTAR_TREES[base]
        cc = CACHE / f"compile/{base}-.venv"
        env.update(MSTAR_SHM_ARENA="1", TORCHINDUCTOR_CACHE_DIR=str(cc / "inductor"), TRITON_CACHE_DIR=str(cc / "triton"),
                   FLASHINFER_WORKSPACE_BASE=str(cc), PATH=f"{tr}/.venv/bin:{env['PATH']}")
        model = env.get("Q3_MODEL", "qwen3_tts_1p7b")
        config = env.get("Q3_CONFIG", "configs/qwen3tts_1p7b.yaml")
        cmd = [f"{tr}/.venv/bin/mstar", "serve", model, "--config", config, "--gpus", str(gpu), "--port", str(port),
               "--socket-path-prefix", f"/tmp/mq3_{base}_g{gpu}"] + extra
        return cmd, env, tr
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if base == "sgomni":
        cfg = env.get("Q3_SGO_CONFIG", str(HARN / "sgomni_qwen3tts_1p7b.yaml"))
        cmd = [f"{ENG}/sgomni/bin/sgl-omni", "serve", "--model-path", HF_ID, "--config", cfg,
               "--host", "127.0.0.1", "--port", str(port)] + extra
        return cmd, env, ENG
    if base == "voxserve":
        cmd = [f"{ENG}/voxserve/.venv/bin/vox-serve", "--model", env.get("Q3_VOX_MODEL", HF_ID), "--host", "127.0.0.1",
               "--port", str(port), "--socket-suffix", f"_q3{port}"] + extra
        return cmd, env, ENG / "voxserve"
    raise SystemExit(f"unknown arm {arm}")


def gpu_used_mb(gpu: int) -> int:
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def port_busy(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def smoke(arm: str, port: int, out: Path) -> None:
    eng = "mstar" if arm.split("+")[0] in MSTAR_TREES else arm.split("+")[0]
    p = subprocess.run([str(CLIENT_PY), str(HARN / "slo_bench.py"), "collect", "--engine", eng, "--url",
                        f"http://127.0.0.1:{port}", "--texts", str(HARN / "smoke.jsonl"), "--concurrency", "1",
                        "--no-warmup", "--out", str(out)], capture_output=True, text=True, timeout=600)
    print(p.stdout.strip()[-400:], p.stderr.strip()[-800:])
    s = json.loads((out / "summary.json").read_text())
    if s["failed"] or not s["ok"] or (s["audio_s_mean"] or 0) < 0.5:
        raise SystemExit(f"smoke failed for {arm}: {s}")


def serve(arm: str, gpu: int, port: int, tag: str, extra: list[str], envs: list[str], timeout: int = 3600) -> dict:
    global STATE
    STATE = state_file(gpu)
    if STATE.exists():
        raise SystemExit(f"a server is already recorded in {STATE}; stop it first")
    if port_busy(port):
        raise SystemExit(f"port {port} busy")
    for _ in range(60):
        if gpu_used_mb(gpu) < 2048:
            break
        time.sleep(5)
    else:
        raise SystemExit(f"GPU {gpu} holds {gpu_used_mb(gpu)} MB")
    for kv in envs:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    cmd, env, cwd = arm_cmd(arm, gpu, port, extra)
    logdir = RES / "servers"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{tag}.log"
    with open(log, "w") as fh:
        fh.write(f"# {time.strftime('%F %T')} {' '.join(cmd)}\n# env: {' '.join(envs)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    state = {"arm": arm, "pid": proc.pid, "pgid": os.getpgid(proc.pid), "port": port, "gpu": gpu, "log": str(log),
             "tag": tag, "cmd": cmd, "envs": envs, "started": t0}
    STATE.write_text(json.dumps(state))
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            STATE.unlink(missing_ok=True)
            raise SystemExit(f"server exited rc={proc.returncode}; see {log}")
        if healthy(port):
            break
        time.sleep(3)
    else:
        stop()
        raise SystemExit(f"not healthy after {timeout}s; see {log}")
    state["healthy_after_s"] = round(time.time() - t0, 1)
    STATE.write_text(json.dumps(state))
    print(f"healthy after {state['healthy_after_s']}s: {arm} pid {proc.pid} port {port} log {log}", flush=True)
    # competitors answer /health before their graphs are captured: retry the smoke until audio comes back
    deadline = time.time() + (2400 if arm.split("+")[0] not in MSTAR_TREES else 0)
    while True:
        try:
            smoke(arm, port, RES / "smoke" / tag)
            break
        except SystemExit as e:
            if time.time() < deadline and proc.poll() is None and not crashed(log):
                print(f"smoke not ready yet ({e}); retrying", flush=True)
                continue
            stop()
            raise
        except BaseException:
            stop()
            raise
    state["ready_after_s"] = round(time.time() - t0, 1)
    STATE.write_text(json.dumps(state))
    with open(RES / "startup.jsonl", "a") as fh:
        fh.write(json.dumps({k: state[k] for k in ("arm", "tag", "healthy_after_s", "ready_after_s", "envs")}) + "\n")
    return state


def crashed(log) -> bool:
    """A server process (or one of its children, e.g. a competitor's scheduler) died with a traceback."""
    try:
        txt = Path(log).read_text(errors="replace")
    except OSError:
        return False
    return "Traceback (most recent call last)" in txt or "OutOfMemoryError" in txt


def stop() -> None:
    if not STATE.exists():
        return
    s = json.loads(STATE.read_text())
    try:
        os.killpg(s["pgid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(60):
        try:  # the server is our child when stop() runs inside a cell: reap it, or its zombie keeps the group alive
            os.waitpid(s["pid"], os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.killpg(s["pgid"], 0)
            time.sleep(1)
        except ProcessLookupError:
            break
    else:
        try:
            os.killpg(s["pgid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    STATE.unlink(missing_ok=True)
    for _ in range(60):  # let the GPU free before the next arm
        if gpu_used_mb(s["gpu"]) < 2048:
            break
        time.sleep(2)
    print(f"stopped {s['arm']} ({s['tag']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve", "stop", "cell"])
    ap.add_argument("arm", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("sub", nargs="?")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--serve-arg", action="append", default=[])
    ap.add_argument("--keep", action="store_true", help="cell: leave the server running")
    args, rest = ap.parse_known_args()
    global STATE
    STATE = state_file(args.gpu)
    if rest and rest[0] == "--":
        rest = rest[1:]
    if args.cmd == "stop":
        stop()
        return
    tag = args.tag or f"{args.arm}_{time.strftime('%H%M%S')}"
    if args.cmd == "serve":
        serve(args.arm, args.gpu, args.port, tag, args.serve_arg, args.env)
        return
    # cell: one server, one slo_bench invocation (sweep/closed/collect), then stop
    eng = "mstar" if args.arm.split("+")[0] in MSTAR_TREES else args.arm.split("+")[0]
    out = RES / args.sub
    out.mkdir(parents=True, exist_ok=True)
    (out / "cell.json").write_text(json.dumps({"arm": args.arm, "tag": tag, "gpu": args.gpu, "env": args.env, "serve_args": args.serve_arg,
                                               "bench": rest, "commit": commit_of(args.arm)}, indent=1))
    try:
        serve(args.arm, args.gpu, args.port, tag, args.serve_arg, args.env)
        cmd = [str(CLIENT_PY), str(HARN / "slo_bench.py"), args.kind, "--engine", eng, "--url",
               f"http://127.0.0.1:{args.port}", "--out", str(out)] + rest
        print("bench:", " ".join(cmd), flush=True)
        # coarse GPU-busy for every arm (competitors have no phase profile), all GPUs (shows borrowers' overlap), every 500 ms
        smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits", "-lms", "500"], stdout=open(out / "gpu_util.csv", "w"))
        try:
            rc = subprocess.run(cmd).returncode
        finally:
            smi.terminate()
        log = RES / "servers" / f"{tag}.log"
        if eng != "mstar" and crashed(log):  # a mid-run crash leaves numbers that look like overload: never mark such a cell DONE
            print(f"server crashed during the bench (traceback in {log}); cell failed", flush=True)
            rc = rc or 3
        if rc:
            print(f"bench rc={rc}", flush=True)
            return rc
    finally:
        if not args.keep:
            stop()


def commit_of(arm: str) -> str | None:
    base = arm.split("+")[0]
    if base in MSTAR_TREES:
        p = MSTAR_TREES[base] / "COMMIT.txt"
        return p.read_text().strip() if p.exists() else None
    return {"sgomni": "sglang-omni==0.1.8", "voxserve": "vox-serve git ce212de19"}.get(base)


if __name__ == "__main__":
    sys.exit(main())
