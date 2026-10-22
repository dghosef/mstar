"""Overnight video A/B driver (runs on the box).

  python ovn_video_bench.py run --tree /root/work/base --label base_c1 \
      --config configs/wan22.yaml --gpus 0 --port 8100 \
      --size 1280x704 --frames 81 --steps 50 --concurrency 1 --num 3 --warmup 1

Starts one server from --tree (cwd + PYTHONPATH = tree, so that checkout is imported),
always with MSTAR_PHASE_TIMING (the profile pass) and --log-stats-file, waits for /health,
fires warmup + measured requests closed-loop at --concurrency, saves every mp4, stops the
server (SIGINT to its session, then SIGKILL), and writes <out>/<label>/summary.json with
client e2e, throughput, per-node phase times (from the request profiles) and the worker
phase tables.

  python ovn_video_bench.py psnr <dirA> <dirB>     # per-file PSNR of decoded frames
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "a person swimming in ocean",
    "a person giving a presentation to a room full of colleagues",
    "a person washing the dishes",
    "a person eating a burger",
    "a person walking in the snowstorm",
    "a person drinking coffee in a cafe",
    "a person playing guitar",
    "a bicycle leaning against a tree",
    "a bicycle gliding through a snowy field",
    "a bicycle slowing down to stop",
]


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001
        return False


def port_busy(port: int) -> bool:
    out = subprocess.run(["ss", "-ltnH", f"sport = :{port}"], capture_output=True, text=True).stdout
    return bool(out.strip())


def gpu_mem() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip().replace("\n", " | ")


def start_server(args, log_path: str, stats_path: str) -> subprocess.Popen:
    if port_busy(args.port):
        raise SystemExit(f"port {args.port} busy; refusing to start")
    env = dict(os.environ)
    env["PYTHONPATH"] = args.tree
    env["MSTAR_PHASE_TIMING"] = str(args.phase_period)
    env["MSTAR_SHM_ARENA"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    venv_bin = os.path.dirname(args.python)
    env["PATH"] = venv_bin + ":" + env.get("PATH", "")
    env["VIRTUAL_ENV"] = os.path.dirname(venv_bin)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    cmd = [args.python, "-m", "mstar.cli.main", "serve", args.model, "--gpus", args.gpus,
           "--port", str(args.port), "--log-stats-file", stats_path]
    if args.config:
        cmd += ["--config", args.config]
    if args.nsys:
        cmd = ["nsys", "launch", f"--session-new={NSYS_SESSION}", "--trace=cuda,nvtx", "--wait=all"] + cmd
    open(stats_path, "w").close()
    log = open(log_path, "w")
    log.write(f"# cmd: {' '.join(cmd)}\n# tree: {args.tree}\n# env: {args.env}\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=args.tree, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    deadline = time.time() + args.start_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited rc={proc.returncode}; see {log_path}")
        if healthy(args.port):
            return proc
        time.sleep(3)
    stop_server(proc)
    raise SystemExit("server start timeout")


NSYS_SESSION = "ovnvid"


def descendants(root: int) -> list[int]:
    ps = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    parent = {}
    for line in ps.splitlines():
        pid, ppid = (int(x) for x in line.split())
        parent[pid] = ppid
    out = []
    for pid in parent:
        q = pid
        while q in parent and q != root and q > 1:
            q = parent[q]
        if q == root and pid != root:
            out.append(pid)
    return out


def stop_server(proc: subprocess.Popen) -> None:
    kids = descendants(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        pass
    # reap the whole session (workers are multiprocessing spawns)
    ps = subprocess.run(["ps", "-e", "-o", "pid=,sid="], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        pid, sid = line.split()
        if int(sid) == proc.pid:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in kids:  # wrappers like `nsys launch` start the server in another session
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for _ in range(60):
        if not port_busy(proc._port):  # type: ignore[attr-defined]
            break
        time.sleep(1)
    time.sleep(5)


def gen_one(args, idx: int, save_dir: str | None) -> dict:
    prompt = PROMPTS[idx % len(PROMPTS)]
    payload = {
        "prompt": prompt, "negative_prompt": args.negative, "size": args.size, "seed": idx,
        "guidance_scale": args.guidance, "num_inference_steps": args.steps,
        "num_frames": args.frames, "fps": 24,
    }
    if args.model_name:
        payload["model"] = args.model_name
    if args.image:
        with open(args.image, "rb") as f:
            payload["image"] = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/videos/generations",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.req_timeout) as r:
            out = json.load(r)
        dt = time.perf_counter() - t0
        mp4 = base64.b64decode(out["data"][0]["b64_json"])
        if save_dir:
            with open(os.path.join(save_dir, f"{idx:03d}.mp4"), "wb") as f:
                f.write(mp4)
        return {"idx": idx, "ok": True, "e2e_s": dt, "bytes": len(mp4), "t_end": time.time()}
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "ok": False, "e2e_s": time.perf_counter() - t0, "error": repr(e)[:300]}


def run_wave(args, indices: list[int], save_dir: str | None) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(lambda i: gen_one(args, i, save_dir), indices))
    return res, time.perf_counter() - t0


def node_phases(stats_text: str, tree: str) -> list[dict]:
    sys.path.insert(0, tree)
    from benchmark.wan22.profile_log import parse_profiles
    out = []
    for p in parse_profiles(stats_text):
        out.append({
            "rid": p.rid, "total_ms": p.total_ms, "timeline_ms": dict(p.timeline_ms),
            "nodes": {n: {"n": t.exec_count, "total_ms": t.total_ms, "fwd_ms": t.fwd_ms}
                      for n, t in p.nodes.items()},
        })
    return out


def worker_phase_tables(log_text: str, tree: str) -> str:
    sys.path.insert(0, tree)
    try:
        from benchmark.worker_phases.phases import parse_log, render, segment
        return render(segment(parse_log(log_text)))
    except Exception as e:  # noqa: BLE001
        return f"(phase parse failed: {e!r})"


def cmd_run(args) -> None:
    out_dir = os.path.join(args.out, args.label)
    os.makedirs(out_dir, exist_ok=True)
    save_dir = os.path.join(out_dir, "videos")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "server.log")
    stats_path = os.path.join(out_dir, "stats.txt")
    t_start = time.time()
    proc = start_server(args, log_path, stats_path)
    proc._port = args.port  # type: ignore[attr-defined]
    startup_s = time.time() - t_start
    mem_ready = gpu_mem()
    peak = {"mem": mem_ready}
    stop_flag = threading.Event()

    def poll_mem():
        best = 0
        while not stop_flag.is_set():
            s = gpu_mem()
            tot = sum(int(x.split(",")[1].split()[0]) for x in s.split("|") if "," in x)
            if tot > best:
                best, peak["mem"] = tot, s
            time.sleep(1.0)

    th = threading.Thread(target=poll_mem, daemon=True)
    th.start()
    try:
        warm_res = []
        for w in range(args.warmup):
            r, _ = run_wave(args, [900 + w * args.concurrency + j for j in range(args.concurrency)], None)
            warm_res += r
        stats_off = os.path.getsize(stats_path)
        if args.nsys:
            subprocess.run(["nsys", "start", f"--session={NSYS_SESSION}", "-o", os.path.join(out_dir, "prof"),
                            "--force-overwrite=true"], check=False)
        res, wall = run_wave(args, list(range(args.num)), save_dir)
        if args.nsys:
            subprocess.run(["nsys", "stop", f"--session={NSYS_SESSION}"], check=False)
    finally:
        stop_flag.set()
        with open(log_path) as f:
            log_text = f.read()
        stop_server(proc)
    with open(stats_path) as f:
        f.seek(stats_off)
        stats_text = f.read()
    oks = [r for r in res if r["ok"]]
    e2e = [r["e2e_s"] for r in oks]
    summary = {
        "label": args.label, "tree": args.tree, "config": args.config, "env": args.env,
        "size": args.size, "frames": args.frames, "steps": args.steps,
        "guidance": args.guidance, "concurrency": args.concurrency, "num": args.num,
        "startup_s": startup_s, "failed": len(res) - len(oks),
        "warmup_failed": sum(1 for r in warm_res if not r["ok"]),
        "warmup_e2e_s": [r["e2e_s"] for r in warm_res],
        "e2e_s": e2e, "e2e_mean_s": statistics.mean(e2e) if e2e else None,
        "e2e_median_s": statistics.median(e2e) if e2e else None,
        "wall_s": wall, "videos_per_min": 60 * len(oks) / wall if wall else None,
        "peak_gpu_mem": peak["mem"],
        "requests": res,
        "node_phases": node_phases(stats_text, args.tree),
    }
    errs = [r.get("error") for r in res if not r["ok"]]
    if errs:
        summary["errors"] = errs[:5]
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    tables = worker_phase_tables(log_text, args.tree)
    with open(os.path.join(out_dir, "worker_phases.txt"), "w") as f:
        f.write(tables)
    # compact console line
    np_ = summary["node_phases"]
    agg: dict[str, list[float]] = {}
    for p in np_:
        for n, t in p["nodes"].items():
            agg.setdefault(n, []).append(t["total_ms"] / max(1, 1))
            if n == "dit" and t["n"]:
                agg.setdefault("dit_step", []).append(t["total_ms"] / t["n"])
    post = [p["timeline_ms"].get("conductor done -> first chunk") for p in np_]
    post = [x for x in post if x is not None]
    if post:
        agg["post"] = post
    nodes_s = " ".join(f"{n}={statistics.mean(v):.0f}ms" for n, v in agg.items())
    print(f"RESULT {args.label}: failed={summary['failed']} e2e_mean={summary['e2e_mean_s']}"
          f" wall={wall:.1f}s vpm={summary['videos_per_min']:.3f} {nodes_s} startup={startup_s:.0f}s",
          flush=True)


def decode_frames(path: str):
    import av
    import numpy as np
    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def psnr(a, b) -> float:
    import numpy as np
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def cmd_psnr(args) -> None:
    res = {}
    for name in sorted(os.listdir(args.a)):
        pa, pb = os.path.join(args.a, name), os.path.join(args.b, name)
        if not name.endswith(".mp4") or not os.path.exists(pb):
            continue
        fa, fb = decode_frames(pa), decode_frames(pb)
        if fa.shape != fb.shape:
            res[name] = f"shape {fa.shape} vs {fb.shape}"
            continue
        res[name] = round(psnr(fa, fb), 2)
    print(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", default="/workspace/results")
    r.add_argument("--model", default="wan22")
    r.add_argument("--model-name", default="", help="'model' field in the request body")
    r.add_argument("--config", default="")
    r.add_argument("--gpus", default="0")
    r.add_argument("--port", type=int, default=8100)
    r.add_argument("--python", default="/root/work/mstar/.venv/bin/python")
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--phase-period", type=int, default=20)
    r.add_argument("--nsys", action="store_true", help="nsys-capture the measured wave to <label>/prof.nsys-rep")
    r.add_argument("--size", default="1280x704")
    r.add_argument("--frames", type=int, default=81)
    r.add_argument("--steps", type=int, default=50)
    r.add_argument("--guidance", type=float, default=5.0)
    r.add_argument("--negative", default="")
    r.add_argument("--image", default="", help="i2v conditioning image (png)")
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--num", type=int, default=3)
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--start-timeout", type=int, default=2400)
    r.add_argument("--req-timeout", type=int, default=3600)
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("psnr")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_psnr)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
