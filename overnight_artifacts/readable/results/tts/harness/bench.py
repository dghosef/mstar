#!/usr/bin/env python3
"""ovn-tts bench driver: start an mstar server from a tree, run TTS closed-loop points, stop it.

    python bench.py --tree /root/work/mstar --label kokoro_base --model kokoro \
        --points 1:s26:1,32:s156:2 --phase 100 --gpus 0 --port 8100

Each point is concurrency:sentence_file_stem:repeats. Results under /workspace/results/<label>/.
The phase pass (MSTAR_PHASE_TIMING) is parsed from the server log into phases.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
SERVER_SH = ".claude/skills/benchmarking/scripts/server.sh"
PY = "/root/work/mstar/.venv/bin/python"


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True, help="mstar serve model name")
    ap.add_argument("--req-model", default=None, help="model field in the request")
    ap.add_argument("--config", default=None)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--points", default="1:s26:1,32:s156:2")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL for the server")
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--out-root", default="/workspace/results")
    ap.add_argument("--spy", default=None, help="point name (e.g. c32_s156) to py-spy during, after --spy-delay s")
    ap.add_argument("--spy-delay", type=float, default=20.0)
    ap.add_argument("--spy-seconds", type=int, default=12)
    ap.add_argument("--mps", action="store_true", help="run under a CUDA MPS daemon on --gpus (one GPU)")
    ap.add_argument("--asr", action="store_true", help="points are conc:num_requests:_ for benchmark.asr_eval")
    args = ap.parse_args()

    out = Path(args.out_root) / args.label
    out.mkdir(parents=True, exist_ok=True)
    tree = Path(args.tree)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(tree), "MSTAR_SHM_ARENA": "1", "HF_HOME": "/root/work/hf",
        "PATH": f"/root/work/mstar/.venv/bin:{env.get('PATH', '')}",
        "VIRTUAL_ENV": "/root/work/mstar/.venv",
        "SERVER_STATE_DIR": f"/root/work/servers_{args.port}", "USER": "root",
        "TORCHINDUCTOR_CACHE_DIR": "/root/work/inductor_cache", "TORCHINDUCTOR_COMPILE_THREADS": "8", "MAX_JOBS": "16",
    })
    if args.phase:
        env["MSTAR_PHASE_TIMING"] = str(args.phase)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = out / "server.log"
    gpus = args.gpus
    if args.mps:
        d = f"/tmp/mps_root_gpu{args.gpus}"
        run(["bash", str(tree / SERVER_SH), "mps-start", args.gpus], cwd=tree, env=env)
        env.update({"CUDA_MPS_PIPE_DIRECTORY": f"{d}/pipe", "CUDA_MPS_LOG_DIRECTORY": f"{d}/log"})
        gpus = "0"
    serve = [PY, "-m", "mstar.cli.main", "serve", args.model, "--gpus", gpus, "--port", str(args.port)]
    if args.config:
        serve += ["--config", args.config]
    meta = {"label": args.label, "tree": str(tree), "serve": serve, "env": {k: env[k] for k in env if k.startswith(("MSTAR", "TORCH"))},
            "points": args.points, "started": time.strftime("%F %T"),
            "load_start": open("/proc/loadavg").read().split()[:3]}
    imp = subprocess.run([PY, "-c", "import mstar; print(mstar.__file__)"], cwd=tree, env=env,
                         capture_output=True, text=True)
    meta["mstar_file"] = imp.stdout.strip()
    t0 = time.time()
    r = run(["bash", str(tree / SERVER_SH), "start", args.label, str(args.port), "--timeout", str(args.timeout),
             "--log", str(log), "--", *serve], cwd=tree, env=env)
    meta["startup_s"] = time.time() - t0
    if r.returncode != 0:
        meta["error"] = "server failed to start"
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        run(["bash", str(tree / SERVER_SH), "stop", args.label], cwd=tree, env=env)
        sys.exit(1)
    try:
        summaries = []
        for pt in args.points.split(","):
            conc, stem, reps = pt.split(":")
            name = f"c{conc}_{stem}"
            while any(x.get("point") == name for x in summaries):
                name += "_again"
            if args.asr:
                cmd = [PY, "-m", "benchmark.asr_eval", "--url", f"http://127.0.0.1:{args.port}", "--model", args.req_model or args.model,
                       "--num-requests", stem, "--max-concurrency", conc, "--num-warmup", str(max(4, 2 * int(conc))),
                       "--local-cache", "/root/work/asr_cache", "--output-json", str(out / f"{name}.json")]
                with open(out / f"{name}.log", "w") as fh:
                    run(cmd, cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT)
                try:
                    rep = json.loads((out / f"{name}.json").read_text())
                    summaries.append({"point": name, "wall_s": rep["wall_time_s"], "wer": rep["wer"],
                                      "req_s": rep["num_requests"] / rep["wall_time_s"], "errors": len(rep["errors"])})
                    print(summaries[-1], flush=True)
                except Exception as exc:  # noqa: BLE001
                    summaries.append({"point": name, "error": str(exc)})
                continue
            cmd = [PY, "-m", "benchmark.tts_speech_bench", "--engine", "mstar",
                   "--url", f"http://127.0.0.1:{args.port}", "--model", args.req_model or args.model,
                   "--sentences", str(HARNESS / f"{stem}.txt"), "--concurrency", conc,
                   "--repeats", reps, "--warmup", str(max(4, 2 * int(conc))),
                   "--out", str(out / f"{name}.json"), "--label", args.label]
            if not args.no_audio:
                cmd += ["--save-audio-dir", str(out / f"{name}_wav")]
            if args.voice:
                cmd += ["--voice", args.voice]
            for e in args.extra:
                cmd += ["--extra", e]
            spy = None
            if args.spy == name:
                spy = subprocess.Popen(
                    f"sleep {args.spy_delay}; {PY} {HARNESS / 'spy.py'} --port {args.port} --seconds {args.spy_seconds} "
                    f"--out {out / 'spy'} > {out / 'spy.txt'} 2>&1", shell=True)
            with open(out / f"{name}.log", "w") as fh:
                run(cmd, cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT)
            if spy is not None:
                spy.wait()
            try:
                rep = json.loads((out / f"{name}.json").read_text())
                summaries.append({"point": name, **rep["median_over_repeats"], "errors": rep["errors_total"]})
                print(rep["markdown_row"], flush=True)
            except Exception as exc:  # noqa: BLE001
                summaries.append({"point": name, "error": str(exc)})
        meta["summaries"] = summaries
    finally:
        run(["bash", str(tree / SERVER_SH), "stop", args.label], cwd=tree, env=env)
        if args.mps:
            time.sleep(3)
            run(["bash", str(tree / SERVER_SH), "mps-stop", args.gpus], cwd=tree, env=env)
    if args.phase:
        code = ("import sys; from benchmark.worker_phases.phases import parse_log, segment, render;"
                "print(render(segment(parse_log(open(sys.argv[1]).read()))))")
        with open(out / "phases.txt", "w") as fh:
            subprocess.run([PY, "-c", code, str(log)], cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT)
    meta["load_end"] = open("/proc/loadavg").read().split()[:3]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("BENCH DONE", args.label, "load", meta["load_start"][0], "->", meta["load_end"][0], flush=True)


if __name__ == "__main__":
    main()
