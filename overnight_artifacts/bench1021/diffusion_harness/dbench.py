#!/usr/bin/env python3
"""Diffusion lane of the 2026-10-21/22 publication benchmark: M* (PR d248477ec, main abdfd5802)
vs vllm-omni 0.30.0, SGLang diffusion 0.5.22 and diffusers 0.41.0, on node2 (8x H200).

Library of phases used by /root/work/bench/queue.py (the unattended runner); also a CLI:
  smoke     start one arm on a GPU, one request, stop
  collect   correctness collection, untimed, jobs in parallel on separate GPUs: served arms B=1 over
            the gate slice (prompt i, seed 1000+i), outputs saved; mstar-pr twice (same-code control);
            diffusers yardsticks ref_a/ref_b/ref_k/ref_seed (ref_c after tuning) -- see dgate.py
  tune      competitor knob sweep, one server per knob set on GPU 0 with nothing else on the box:
            B=1 + c=8 (images) / c=1 + c=4 (video), then that knob set's gate outputs (untimed)
  gate      dgate.py over everything collected (criteria fixed in dgate.py's docstring)
  select    tuned config per competitor = best tune score among knob sets whose gate passed
  rounds    ROUNDS x arms on GPU 0, one server on the box; each round restarts every server and
            rotates the arm order. Per server: smoke, one untimed pass over the prompt set, then per
            point 1 warmup wave + the measured requests. diffusers in-process (offline batches b = c).
  profile   mstar-pr under MSTAR_PHASE_TIMING=100, rendered with benchmark.worker_phases

Robustness (for unattended runs): every server start and every cell has a hard timeout; a cell that
fails (failed != 0, timeout, crash) is retried once on a fresh server, and the retry's row supersedes
it; cells already in results.jsonl with failed == 0 are skipped; nothing timed starts after DEADLINE.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path("/root/work")
ENG = WORK / "engines"
RESULTS = Path("/workspace/results/bench")
SERVER_SH = HERE / "server.sh"
STATE_DIR = Path(os.environ.get("SERVER_STATE_DIR", "/tmp/servers_root"))
CLIENT_PY = WORK / "clientenv/bin/python"
DIFF_PY = ENG / "diffusers/bin/python"
TREES = {"mstar-pr": WORK / "mstar", "mstar-main": WORK / "mstar-main"}
PARTI = WORK / "data/parti64.txt"
VBENCH = WORK / "data/vbench16.txt"
# no timed work may start after this (13:00 PDT Oct 22 = 20:00 UTC), minus a margin per model kind
DEADLINE = float(os.environ.get("DBENCH_DEADLINE", "1792699200"))  # 2026-10-22 20:00:00 UTC

IMG_NUMS = {1: 32, 2: 16, 4: 32, 8: 48, 16: 80}
MODELS = {
    "zimage": dict(hf="Tongyi-MAI/Z-Image-Turbo", mstar="z_image_turbo", config="configs/z_image_turbo.yaml",
                   kind="image", size="1024x1024", steps=8, guidance=0.0, prompts=PARTI, gate_count=64,
                   concs=(1, 2, 4, 8, 16), nums=IMG_NUMS, warm_conc=8),
    "klein4b": dict(hf="black-forest-labs/FLUX.2-klein-4B", mstar="flux2_klein", config="configs/flux2_klein.yaml",
                    kind="image", size="1024x1024", steps=4, guidance=1.0, prompts=PARTI, gate_count=64,
                    concs=(1, 2, 4, 8, 16), nums=IMG_NUMS, warm_conc=8),
    "klein9b": dict(hf="black-forest-labs/FLUX.2-klein-9B", mstar="flux2_klein_9b",
                    config="configs/flux2_klein_9b.yaml", kind="image", size="1024x1024", steps=4, guidance=1.0,
                    prompts=PARTI, gate_count=64, concs=(1, 2, 4, 8, 16), nums=IMG_NUMS, warm_conc=8),
    "wan22": dict(hf="Wan-AI/Wan2.2-TI2V-5B-Diffusers", mstar="wan22", config="configs/wan22.yaml",
                  kind="video", size="1280x704", steps=20, guidance=5.0, frames=81, flow_shift=5.0, prompts=VBENCH,
                  gate_count=4, concs=(1, 2, 4, 8), nums={1: 3, 2: 4, 4: 8, 8: 16}, warm_conc=1, long_steps=50,
                  long_num=3),
}
KNOBS = {"vllmomni": {"base": []}, "sglang": {"base": ["--performance-mode", "speed"]},
         "diffusers": {"eager": ["--compile", "none"]}}
PORT0 = 8400

# what the runner is doing right now (queue.py's heartbeat thread prints it)
# (kept across importlib.reload, which re-runs this module in the same globals)
STATUS = globals().get("STATUS") or {"model": None, "phase": None, "arm": None, "cell": None}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def loadavg():
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return None


def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


def wait_gpus_free(gpus=None, timeout=600):
    t0 = time.time()
    while True:
        used = gpu_mem()
        sel = [used[g] for g in gpus] if gpus is not None else used
        if max(sel) <= 2048:
            return True
        if time.time() - t0 > timeout:
            return False
        time.sleep(5)


def reap_all(reason=""):
    """Stop every server this harness started (server.sh state), then kill any process still
    holding GPU memory. The node is dedicated to this lane, so every GPU process is ours."""
    for pid_file in STATE_DIR.glob("*.pid"):
        subprocess.run([str(SERVER_SH), "stop", pid_file.stem], capture_output=True, timeout=300)
    if wait_gpus_free(timeout=60):
        return
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    pids = [int(x) for x in out.split() if x.strip().isdigit()]
    log(f"reap ({reason}): GPU still held by {pids}; SIGINT then SIGKILL")
    for sig in (signal.SIGINT, signal.SIGKILL):
        for p in pids:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        if wait_gpus_free(timeout=60):
            return


def tail(path, n=20):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def failure(model, arm, phase, cell, error, action):
    """One line per FINAL failure (after its retry) -> RESULTS/failures.jsonl, watched from the Mac."""
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "model": model, "arm": arm,
           "phase": phase, "cell": cell, "error": str(error)[-4000:], "action_taken": action}
    with (RESULTS / "failures.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    log(f"FAILURE {model} {arm} {phase} {cell}: {action}")


FATAL_LOG = ("illegal memory access", "CUDA error", "Traceback")


def gpu_util(gpu=0):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def descendants(root):
    """{pid: cmdline} of long-lived python processes below root (inductor compile workers excluded)."""
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    parent, args = {}, {}
    for line in out.splitlines():
        f = line.split(None, 2)
        if len(f) >= 2:
            parent[int(f[0])], args[int(f[0])] = int(f[1]), f[2] if len(f) > 2 else ""
    res = {}
    for pid in parent:
        q = pid
        while q in parent and q != root and q > 1:
            q = parent[q]
        if q == root and "python" in args[pid] and "compile_worker" not in args[pid]:
            res[pid] = args[pid]
    return res


def run_watched(cmd, timeout, watch=None, **kw):
    """subprocess.run with a liveness watchdog, polled every 30 s. Aborts at once when the server's health
    endpoint stops answering, one of its long-lived child processes exits, its log gains a fatal line
    (Traceback / CUDA error / illegal memory access), or GPU 0 stays at 0% for longer than watch['idle_s'].
    Returns (returncode or None, abort reason or None)."""
    import urllib.request
    watch = watch or {}
    p = subprocess.Popen(cmd, **kw)
    t0 = time.time()
    log_path = watch.get("log")
    log_off = Path(log_path).stat().st_size if log_path and Path(log_path).exists() else 0
    pid = None
    if watch.get("name"):
        try:
            pid = int((STATE_DIR / f"{watch['name']}.pid").read_text())
        except (OSError, ValueError):
            pid = None
    kids0 = descendants(pid) if pid else {}
    idle_since = None
    reason = None
    while True:
        try:
            p.wait(timeout=30)
            return p.returncode, None
        except subprocess.TimeoutExpired:
            pass
        if time.time() - t0 > timeout:
            reason = f"timeout {timeout}s"
        if not reason and watch.get("port"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{watch['port']}{watch.get('health', '/health')}",
                                            timeout=20) as r:
                    if not 200 <= r.status < 300:
                        reason = f"health {r.status}"
            except Exception as e:  # noqa: BLE001
                reason = f"health endpoint not answering ({e!r})"[:200]
        if not reason and pid:
            gone = set(kids0) - set(descendants(pid))
            if gone:
                reason = "server child process exited: " + "; ".join(kids0[g][:80] for g in sorted(gone))
        if not reason and log_path and Path(log_path).exists():
            with open(log_path, "rb") as f:
                f.seek(log_off)
                new = f.read().decode(errors="replace")
            log_off += len(new.encode())
            hit = next((k for k in FATAL_LOG if k in new), None)
            if hit:
                i = new.find(hit)
                reason = f"server log: {hit}: {new[max(0, i - 300):i + 300]}"
        if not reason and watch.get("idle_s"):
            u = gpu_util(0)
            if u == 0:
                idle_since = idle_since or time.time()
                if time.time() - idle_since > watch["idle_s"]:
                    reason = f"GPU 0 idle for {int(time.time() - idle_since)}s"
            else:
                idle_since = None
        if reason:
            p.kill()
            p.wait()
            log(f"!! WATCHDOG abort: {reason[:300]}")
            return None, reason


def setup_diffusion(b):
    """Idempotent: engine venvs (install_engines.sh skips what exists), the client env, the model's weights,
    the prompt files. Raises if something the entry needs is still missing afterwards."""
    env = {**os.environ, "HF_HOME": str(WORK / "hf"), "PATH": f"/root/.local/bin:{os.environ['PATH']}"}
    tok = WORK / "hf/token"
    if tok.exists():
        env["HF_TOKEN"] = tok.read_text().strip()
    subprocess.run(["bash", str(HERE / "install_engines.sh")], env=env, timeout=3600,
                   stdout=open(b.out / "setup.log", "a"), stderr=subprocess.STDOUT)
    code = f"from huggingface_hub import snapshot_download as s; print(s({b.m['hf']!r}))"
    r = subprocess.run([str(TREES["mstar-pr"] / ".venv/bin/python"), "-c", code], env=env, capture_output=True,
                       text=True, timeout=7200)
    missing = [str(p) for p in (CLIENT_PY, DIFF_PY, ENG / "vllmomni/bin/vllm", ENG / "sgld/bin/sglang",
                                Path(b.m["prompts"]), TREES["mstar-pr"] / ".setup_done",
                                TREES["mstar-main"] / ".setup_done") if not Path(p).exists()]
    if r.returncode != 0 or missing:
        raise RuntimeError(f"setup incomplete: weights rc={r.returncode} {r.stderr[-300:]} missing={missing}")
    log(f"setup ok: {b.key} weights at {r.stdout.strip()}")


class Bench:
    def __init__(self, model, out=None):
        self.key, self.m = model, MODELS[model]
        self.out = Path(out) if out else RESULTS / model
        for d in ("stages", "servers", "cells", "gate"):
            (self.out / d).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- bookkeeping
    def decision(self, kind, **kw):
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": self.key, "kind": kind, **kw}
        with (RESULTS / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"DECISION {kind}: {json.dumps(kw)[:400]}")

    def rows(self):
        p = self.out / "results.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def done_cells(self, tag):
        return {(r["arm"], r.get("knob"), r["conc"], r["round"], r.get("steps")) for r in self.rows()
                if r.get("tag") == tag and r.get("failed") == 0 and "conc" in r}

    def append_row(self, row):
        with (self.out / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def marker(self, n):
        return self.out / "stages" / f"{n}.done"

    def mark_knob_failed(self, arm, kname, why):
        p = self.out / "knob_failed.json"
        d = json.loads(p.read_text()) if p.exists() else {}
        d[f"{arm}__{kname}"] = why
        p.write_text(json.dumps(d, indent=1))

    def knob_failed(self):
        p = self.out / "knob_failed.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def tuned(self):
        p = self.out / "tuned.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def knob_table(self, engine):
        extra = json.loads((self.out / "knobs.json").read_text()) if (self.out / "knobs.json").exists() else {}
        return {**KNOBS.get(engine, {}), **extra.get(engine, {})}

    def knobs(self, engine, name=None):
        table = self.knob_table(engine)
        if name is None:
            name = self.tuned().get(engine, {}).get("config") or next(iter(table))
        return name, table[name]

    # ---------------------------------------------------------------- servers
    def server_cmd(self, arm, gpu, port, args, env_extra=None):
        env = dict(os.environ)
        env.update(HF_HOME=str(WORK / "hf"), HF_HUB_OFFLINE="1")
        env["PATH"] = f"/root/.local/bin:{env['PATH']}"
        if arm in TREES:
            t = TREES[arm]
            env["PATH"] = f"{t}/.venv/bin:{env['PATH']}"
            env.update(MSTAR_SHM_ARENA="1", TORCHINDUCTOR_CACHE_DIR=str(WORK / f"inductor/{arm}"))
            if self.m["kind"] == "image":
                env["MSTAR_REQUIRE_ACCELERATOR_GRAPHS"] = "1"
            if self.key == "wan22":
                # mstar's wan22 loads the UMT5 tokenizer with AutoTokenizer(subfolder="tokenizer"), whose
                # AutoConfig probe has no cached config.json and fails under HF_HUB_OFFLINE=1
                env.pop("HF_HUB_OFFLINE")
            cmd = [f"{t}/.venv/bin/python", "-m", "mstar.cli.main", "serve", self.m["mstar"],
                   "--config", str(t / self.m["config"]), "--gpus", str(gpu), "--port", str(port)] + list(args)
            cwd = t
        elif arm == "vllmomni":
            v = ENG / "vllmomni"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), TORCHINDUCTOR_CACHE_DIR=str(WORK / "inductor/vllmomni"))
            cmd = [f"{v}/bin/vllm", "serve", self.m["hf"], "--omni", "--host", "127.0.0.1", "--port", str(port)] + \
                list(args)
            cwd = ENG
        elif arm == "sglang":
            v = ENG / "sgld"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), TORCHINDUCTOR_CACHE_DIR=str(WORK / "inductor/sglang"))
            cmd = [f"{v}/bin/sglang", "serve", "--model-path", self.m["hf"], "--num-gpus", "1",
                   "--host", "127.0.0.1", "--port", str(port), "--scheduler-port", str(port + 100)] + list(args)
            cwd = ENG
        else:
            raise ValueError(arm)
        if env_extra:
            env.update(env_extra)
        return cmd, env, cwd

    def start(self, arm, label, gpu=0, port=PORT0, args=(), env_extra=None, name=None):
        name = name or f"d{port}"
        cmd, env, cwd = self.server_cmd(arm, gpu, port, args, env_extra)
        slog = self.out / "servers" / f"{label}.log"
        (self.out / "servers" / f"{label}.cmd").write_text(shlex.join(cmd) + "\n" + json.dumps(
            {k: env[k] for k in env if k.startswith(("MSTAR_", "CUDA_VISIBLE", "TORCHINDUCTOR", "SGLANG", "VLLM",
                                                     "HF_HUB_OFFLINE"))}) + "\n")
        limit = 3600 if arm in TREES else 1800
        log(f"start {label} gpu{gpu}:{port} (timeout {limit}s): {shlex.join(cmd)}")
        t0 = time.time()
        try:
            r = subprocess.run([str(SERVER_SH), "start", name, str(port), "--timeout", str(limit), "--log", str(slog),
                                "--", *cmd], env=env, cwd=cwd, capture_output=True, text=True, timeout=limit + 120)
            ok = r.returncode == 0
            err = r.stderr[-500:]
        except subprocess.TimeoutExpired:
            ok, err = False, "server.sh start timed out"
        secs = round(time.time() - t0, 1)
        with (self.out / "startup.jsonl").open("a") as f:
            f.write(json.dumps({"label": label, "arm": arm, "seconds": secs, "ok": ok, "time": time.time()}) + "\n")
        if not ok:
            log(f"!! {label}: start failed after {secs}s: {err}")
            self.stop(name)
            return None
        log(f"   {label} healthy after {secs}s")
        self.live = {"name": name, "port": port, "log": str(slog), "health": "/health"}
        return name

    def stop(self, name):
        self.live = None
        try:
            subprocess.run([str(SERVER_SH), "stop", name], capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            log(f"!! stop {name} timed out")

    def versions(self, arm):
        if arm in TREES:
            t = TREES[arm]
            code = "import mstar, torch, diffusers; print(mstar.__file__, torch.__version__, diffusers.__version__)"
            return {"imports": subprocess.run([f"{t}/.venv/bin/python", "-c", code], capture_output=True, text=True,
                                              cwd=t).stdout.strip(),
                    "commit": (t / "COMMIT.txt").read_text().strip()}
        venv = {"vllmomni": "vllmomni", "sglang": "sgld", "diffusers": "diffusers"}[arm]
        uv = shutil.which("uv") or "/usr/local/bin/uv"
        try:
            fr = subprocess.run([uv, "pip", "freeze", "--python", str(ENG / venv / "bin/python")],
                                capture_output=True, text=True, timeout=120).stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"packages": [], "error": repr(e)}
        keep = ("vllm", "sglang", "torch=", "diffusers", "flashinfer", "transformers", "flash", "triton", "fa3",
                "kernels", "cache-dit", "sgl")
        return {"packages": [x for x in fr.splitlines() if x.lower().startswith(keep)]}

    # ---------------------------------------------------------------- client
    def cell_timeout(self, conc, num, warm, steps=None):
        """Hard wall-clock cap for one client run: generous per-request budget x request waves."""
        per = 60 if self.m["kind"] == "image" else 240 * (steps or self.m["steps"]) / 20
        waves = math.ceil((num + warm * conc) / max(conc, 1)) * (2 if conc > 1 else 1)
        return int(300 + per * waves * (1 + conc / 4))

    def client(self, engine, port, out, conc, num, warm, seed_base=1000, offset=0, save=None, steps=None, live=None):
        m = self.m
        eng = "mstar" if engine in TREES else engine
        req_timeout = 600 if m["kind"] == "image" else 1800
        cmd = [str(CLIENT_PY), "-I", str(HERE / "dclient.py"), "--engine", eng, "--kind", m["kind"], "--port", str(port),
               "--model-id", m["hf"], "--prompts", str(m["prompts"]), "--size", m["size"],
               "--steps", str(steps or m["steps"]), "--guidance", str(m["guidance"]), "--conc", str(conc),
               "--num", str(num), "--warmup-waves", str(warm), "--seed-base", str(seed_base),
               "--prompt-offset", str(offset), "--timeout", str(req_timeout), "--out", str(out)]
        if m["kind"] == "video":
            cmd += ["--frames", str(m["frames"])]
        if m.get("flow_shift") is not None:
            cmd += ["--flow-shift", str(m["flow_shift"])]
        if save:
            cmd += ["--save-dir", str(save)]
        Path(out).mkdir(parents=True, exist_ok=True)
        rf = Path(out) / "results.json"
        if rf.exists():
            rf.unlink()
        limit = self.cell_timeout(conc, num, warm, steps)
        return self.watched_client(cmd, out, limit, idle_s=900 if warm == 0 else 300, live=live)

    def watched_client(self, cmd, out, limit, idle_s, live=None):
        """Run a client under the liveness watchdog against the live server; returns its summary or a failed
        stub carrying the abort reason."""
        live = live if live is not None else (getattr(self, "live", None) or {})
        watch = {**live, "idle_s": idle_s if live.get("port", PORT0) == PORT0 else None}
        with open(Path(out) / "client.log", "w") as f:
            rc, reason = run_watched(cmd, limit, watch, stdout=f, stderr=subprocess.STDOUT, text=True)
        if reason:
            self.last_abort = reason
            return {"failed": None, "error": f"watchdog: {reason}"[:1500]}
        try:
            return json.loads((Path(out) / "results.json").read_text())["summary"]
        except (OSError, ValueError, KeyError):
            return {"failed": None, "error": tail(Path(out) / "client.log", 10)}

    def smoke_request(self, engine, port, live=None):
        s = self.client(engine, port, self.out / "servers" / f"smoke_{engine}_{port}", 1, 1, 0, seed_base=7, live=live)
        if s.get("failed") != 0:
            raise RuntimeError(f"smoke failed: {s}")
        return s

    # ---------------------------------------------------------------- diffusers
    def dref(self, mode, out, gpu, knob_args, extra, log_path, timeout=5400):
        m = self.m
        w, h = m["size"].split("x")
        cmd = [str(DIFF_PY), "-I", str(HERE / "dref.py"), mode, "--model", m.get("dref_model", self.key),
               "--prompts", str(m["prompts"]),
               "--height", h, "--width", w, "--steps", str(m["steps"]), "--guidance", str(m["guidance"]),
               "--out", str(out)] + list(knob_args) + list(extra)
        if m["kind"] == "video":
            cmd += ["--frames", str(m["frames"])]
        # online: diffusers' hub attention kernels (FA3) resolve their version against the Hub
        env = {**os.environ, "HF_HOME": str(WORK / "hf"), "CUDA_VISIBLE_DEVICES": str(gpu),
               "TORCHINDUCTOR_CACHE_DIR": str(WORK / "inductor/diffusers")}
        log(f"dref {mode} gpu{gpu}: {shlex.join(cmd)}")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            rc, reason = run_watched(cmd, timeout, {"log": str(log_path), "idle_s": 1200 if gpu == 0 else None},
                                     env=env, stdout=f, stderr=subprocess.STDOUT)
        if reason:
            with open(log_path, "a") as f:
                f.write(f"\nWATCHDOG abort: {reason}\n")
            return -1
        return rc

    # ---------------------------------------------------------------- phases
    def phase_smoke(self, arm, gpu, knob):
        if arm == "diffusers":
            rc = self.dref("time", self.out / "servers" / "smoke_diffusers", gpu, self.knobs(arm, knob)[1],
                           ["--batch-sizes", "1", "--num", "1"], self.out / "servers" / "smoke_diffusers.log")
            log(f"diffusers smoke rc={rc}")
            return rc == 0
        port = PORT0 + 10 * gpu
        kname, kargs = self.knobs(arm, knob) if arm not in TREES else ("config", [])
        name = self.start(arm, f"smoke_{arm}_{kname}", gpu, port, kargs)
        if not name:
            return False
        try:
            log(f"smoke {arm}: {self.smoke_request(arm, port)}")
            return True
        except Exception as e:  # noqa: BLE001
            log(f"!! smoke {arm}: {e!r}")
            return False
        finally:
            self.stop(name)

    def phase_collect(self, arms, refs, gpus):
        """Untimed: every job on its own GPU, several at once. A job that fails is retried once."""
        gdir = self.out / "gate"
        m = self.m
        jobs = []
        for arm in arms:
            for rep in (("a", "b") if arm == "mstar-pr" else ("a",)):
                jobs.append(("serve", arm, arm.replace("-", "_") + ("_b" if rep == "b" else "")))
        for r in refs:
            jobs.append(("ref", "diffusers", r))
        lock = threading.Lock()
        fpath = gdir / "failed.json"

        def record_failed(tag, n):
            with lock:
                failed = json.loads(fpath.read_text()) if fpath.exists() else {}
                failed[tag] = n
                fpath.write_text(json.dumps(failed))

        def run_once(job, gpu):
            kind, arm, tag = job
            if kind == "ref":
                knob = {"ref_a": ["--attn", "default"], "ref_b": ["--attn", "default"],
                        "ref_k": ["--attn", "_native_efficient"], "ref_seed": ["--attn", "default"]}.get(tag)
                if tag == "ref_c":
                    knob = self.knobs("diffusers")[1]
                extra = ["--count", str(m["gate_count"]), "--seed-base", "1500" if tag == "ref_seed" else "1000"]
                return self.dref("gate", gdir / tag, gpu, knob, extra, gdir / f"{tag}.log") == 0
            port = PORT0 + 10 * gpu
            name = self.start(arm, f"collect_{tag}", gpu, port, [] if arm in TREES else self.knobs(arm)[1],
                              name=f"c{port}")
            if not name:
                return False
            # parallel jobs share this Bench: watch this job's own server (no GPU-idle rule off GPU 0)
            live = {"name": name, "port": port, "log": str(self.out / "servers" / f"collect_{tag}.log"),
                    "health": "/health"}
            try:
                self.smoke_request(arm, port, live=live)
                s = self.client(arm, port, gdir / f"{tag}_cell", 1, m["gate_count"], 0, save=gdir / tag, live=live)
                record_failed(tag, s.get("failed"))
                (gdir / f"{tag}_versions.json").write_text(json.dumps(self.versions(arm)))
                log(f"collect {tag}: failed={s.get('failed')} wall={s.get('wall_s')}")
                return s.get("failed") == 0
            except Exception as e:  # noqa: BLE001
                log(f"!! collect {tag}: {e!r}")
                return False
            finally:
                self.stop(name)

        def run(job, gpu, free):
            tag = job[2]
            try:
                for attempt in (1, 2):
                    if run_once(job, gpu):
                        self.marker(f"collect_{tag}").touch()
                        log(f"collect {tag} done (gpu{gpu}, attempt {attempt})")
                        return
                    log(f"!! collect {tag} attempt {attempt} failed")
                self.decision("collect_failed", tag=tag)
                failure(self.key, tag, "collect", "gate outputs", tail(gdir / f"{tag}.log") or
                        tail(self.out / "servers" / f"collect_{tag}.log"),
                        "gate outputs missing for this tag; its gate verdict will be FAIL")
            finally:
                with lock:
                    free.append(gpu)

        jobs = [j for j in jobs if not self.marker(f"collect_{j[2]}").exists()]
        free, threads = [g for g in gpus], []
        for job in jobs:
            STATUS.update(phase="collect", arm=job[2])
            gpu = None
            while gpu is None:
                with lock:
                    for g in list(free):
                        if wait_gpus_free([g], timeout=1):
                            free.remove(g)
                            gpu = g
                            break
                if gpu is None:
                    time.sleep(5)
            th = threading.Thread(target=run, args=(job, gpu, free))
            th.start()
            threads.append(th)
            time.sleep(5)
        for th in threads:
            th.join()

    def phase_gate(self, arms):
        gdir = self.out / "gate"
        tags = []
        for arm in arms:
            tags.append(arm.replace("-", "_"))
            if arm == "mstar-pr":
                tags.append("mstar_pr_b")
        tags += sorted(d.name for d in gdir.iterdir() if d.is_dir() and "__" in d.name and not d.name.endswith("_cell"))
        cmd = [str(DIFF_PY), "-I", str(HERE / "dgate.py"), "--root", str(gdir), "--kind", self.m["kind"],
               "--prompts", str(self.m["prompts"]), "--count", str(self.m["gate_count"]), "--arms", ",".join(tags),
               "--failed-json", str(gdir / "failed.json"), "--out", str(gdir)]
        env = {**os.environ, "HF_HOME": str(WORK / "hf"), "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "7"}
        with open(gdir / "gate_run.log", "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
        try:
            v = json.loads((gdir / "gate.json").read_text())["verdict"]
        except (OSError, ValueError, KeyError):
            v = {}
        self.decision("gate", verdicts={k: x.get("verdict") for k, x in v.items()})
        if not v:
            failure(self.key, "-", "gate", "-", tail(gdir / "gate_run.log"), "gate did not produce verdicts")
        for k, x in v.items():
            if not str(x.get("verdict", "")).startswith("PASS"):
                failure(self.key, k, "gate", "-", json.dumps(x)[:2000],
                        "knob set ineligible for selection" if "__" in k else
                        "gate FAIL recorded; this arm's timed numbers are not reported as a win")

    def cell(self, tag, arm, knob, conc, rnd, num, warm, port, steps=None):
        st = steps or self.m["steps"]
        d = self.out / "cells" / tag / arm / (f"c{conc}_r{rnd}" if steps is None else f"s{steps}_c{conc}_r{rnd}")
        STATUS.update(cell=f"{tag} {arm}[{knob}] c={conc} r{rnd} steps={st}")
        la0, t0 = loadavg(), time.time()
        s = self.client(arm, port, d, conc, num, warm, seed_base=1000 + 10000 * rnd, steps=steps)
        row = {"tag": tag, "arm": arm, "knob": knob, "conc": conc, "round": rnd, "steps": st, "num": num,
               "load_before": la0, "load_after": loadavg(), "started": t0, "seconds": round(time.time() - t0, 1), **s}
        self.append_row(row)
        unit = "img/s" if self.m["kind"] == "image" else "videos/min"
        thr = s.get("throughput_per_s") if self.m["kind"] == "image" else s.get("throughput_per_min")
        log(f"   {tag} {arm}[{knob}] c={conc} r{rnd} steps={st}: {thr and round(thr, 3)} {unit} "
            f"lat mean {s.get('lat_mean') and round(s['lat_mean'], 3)} failed={s.get('failed')} load={la0}")
        return row

    def served_pass(self, arm, rnd, tag, knob_name=None, concs=None, nums=None, long=True, gate_tag=None):
        """One server process: smoke, warm pass, the cells not yet done; a failed cell is retried once
        on a fresh server process. Returns False if the pass could not complete."""
        m = self.m
        kname, kargs = ("config", []) if arm in TREES else self.knobs(arm, knob_name)
        plan = [(c, (nums or m["nums"])[c], None) for c in (concs or m["concs"])]
        if long and m.get("long_steps"):
            plan.append((1, m["long_num"], m["long_steps"]))
        gate_done = gate_tag is None or (self.out / "gate" / f"{gate_tag}_cell" / "results.json").exists()
        last_err, last_log, last_cell = "", None, "server start"
        for attempt in (1, 2):
            done = self.done_cells(tag)
            todo = [p for p in plan if (arm, kname, p[0], rnd, p[2] or m["steps"]) not in done]
            if not todo and gate_done:
                return True
            if time.time() > DEADLINE:
                self.decision("deadline_skip", tag=tag, arm=arm, round=rnd)
                return False
            STATUS.update(phase=tag, arm=f"{arm}[{kname}] r{rnd} attempt {attempt}", cell="server start")
            reap_all("before start")
            label = f"{tag}_{arm}_{kname}_r{rnd}" + ("" if attempt == 1 else "_retry")
            last_log, last_cell = self.out / "servers" / f"{label}.log", "server start"
            name = self.start(arm, label, 0, PORT0, kargs)
            if not name:
                last_err = "server did not become healthy"
                self.append_row({"tag": tag, "arm": arm, "knob": kname, "round": rnd, "failed": None,
                                 "error": f"server start failed (attempt {attempt})"})
                continue
            try:
                self.smoke_request(arm, PORT0)
                wn = len([p for p in Path(m["prompts"]).read_text().splitlines() if p.strip()]) \
                    if m["kind"] == "image" else 1
                STATUS.update(cell="warm pass")
                ws = self.client(arm, PORT0, self.out / "cells" / tag / arm / f"warmpass_r{rnd}_{kname}",
                                 m["warm_conc"], wn, 0, seed_base=900000)
                log(f"   warm pass {arm}: failed={ws.get('failed')} wall={ws.get('wall_s')}")
                ok = True
                for c, n, steps in todo:
                    last_cell = f"c={c} steps={steps or m['steps']}"
                    r = self.cell(tag, arm, kname, c, rnd, n, 1, PORT0, steps=steps)
                    if r.get("failed") != 0:
                        ok = False
                        last_err = f"cell failed={r.get('failed')} errors={r.get('errors') or r.get('error')}"
                        break  # fresh server for the retry
                (self.out / "versions").mkdir(exist_ok=True)
                (self.out / "versions" / f"{arm}.json").write_text(json.dumps(self.versions(arm)))
                if ok and not gate_done:
                    g = self.out / "gate"
                    STATUS.update(cell=f"gate collect {gate_tag}")
                    s = self.client(arm, PORT0, g / f"{gate_tag}_cell", 1, m["gate_count"], 0, save=g / gate_tag)
                    fp = g / "failed.json"
                    failed = json.loads(fp.read_text()) if fp.exists() else {}
                    failed[gate_tag] = s.get("failed")
                    fp.write_text(json.dumps(failed))
                    gate_done = True
                    log(f"   gate collect {gate_tag}: failed={s.get('failed')}")
                if ok:
                    return True
            except Exception as e:  # noqa: BLE001
                log(f"!! {tag} {arm} r{rnd}: {e!r}")
                last_err = f"pass exception: {e!r}"
                self.append_row({"tag": tag, "arm": arm, "knob": kname, "round": rnd, "failed": None,
                                 "error": f"pass exception: {e!r}"[:300]})
            finally:
                self.stop(name)
                reap_all("after pass")
            if "illegal memory access" in last_err + str(getattr(self, "last_abort", "")):
                self.mark_knob_failed(arm, kname, "illegal memory access")
                break  # a config that corrupts GPU memory is not retried
            if attempt == 1:
                self.decision("retry", tag=tag, arm=arm, knob=kname, round=rnd)
        self.decision("pass_failed", tag=tag, arm=arm, knob=kname, round=rnd)
        failure(self.key, f"{arm}[{kname}]", tag, f"r{rnd} {last_cell}", f"{last_err}\n--- server log tail ---\n"
                f"{tail(last_log) if last_log else ''}",
                "recorded as failed after one retry on a fresh server; queue moves on"
                + (" (knob set ineligible)" if tag.startswith("tune_") else ""))
        return False

    def diffusers_pass(self, rnd, tag, knob_name=None, concs=None, nums=None, long=True):
        m = self.m
        kname, kargs = self.knobs("diffusers", knob_name)
        concs, nums = concs or m["concs"], nums or m["nums"]
        plan = [(c, None) for c in concs] + ([(1, m["long_steps"])] if long and m.get("long_steps") else [])
        for attempt in (1, 2):
            done = self.done_cells(tag)
            todo = [p for p in plan if ("diffusers", kname, p[0], rnd, p[1] or m["steps"]) not in done]
            if not todo:
                return True
            if time.time() > DEADLINE:
                self.decision("deadline_skip", tag=tag, arm="diffusers", round=rnd)
                return False
            reap_all("before diffusers")
            groups = {}
            for c, st in todo:
                groups.setdefault(st, []).append(c)
            for st, cs in groups.items():
                STATUS.update(phase=tag, arm=f"diffusers[{kname}] r{rnd}", cell=f"b={cs} steps={st or m['steps']}")
                d = self.out / "cells" / tag / "diffusers" / (f"r{rnd}_{kname}" + ("" if st is None else f"_s{st}")
                                                               + ("" if attempt == 1 else "_retry"))
                la0, t0 = loadavg(), time.time()
                extra = ["--batch-sizes", ",".join(map(str, cs)),
                         "--num", ",".join(str(nums[c] if st is None else m["long_num"]) for c in cs),
                         "--seed-base", str(1000 + 10000 * rnd)] + (["--steps", str(st)] if st else [])
                rc = self.dref("time", d, 0, kargs, extra, Path(str(d) + ".log"))
                try:
                    rows = json.loads((d / "results.json").read_text())["rows"]
                except (OSError, ValueError, KeyError):
                    rows = [{"batch": c, "num": nums[c], "completed": 0, "failed": None,
                             "error": f"dref rc={rc}, no results"} for c in cs]
                for r in rows:
                    row = {"tag": tag, "arm": "diffusers", "knob": kname, "conc": r["batch"], "round": rnd,
                           "steps": st or m["steps"], "num": r["num"], "load_before": la0, "load_after": loadavg(),
                           "started": t0, "completed": r.get("completed"), "failed": r.get("failed"),
                           "wall_s": r.get("wall_s"), "throughput_per_s": r.get("throughput_per_s"),
                           "throughput_per_min": r.get("throughput_per_min"), "offline_batch": True,
                           "error": r.get("error"), "peak_mem_gib": r.get("peak_mem_gib")}
                    if r["batch"] == 1 and r.get("batch_latencies"):
                        lat = sorted(r["batch_latencies"])
                        row.update(lat_mean=statistics.fmean(lat), lat_p50=statistics.median(lat),
                                   lat_p99=lat[min(len(lat) - 1, round(0.99 * (len(lat) - 1)))], lat_min=lat[0],
                                   lat_max=lat[-1])
                    self.append_row(row)
                    log(f"   {tag} diffusers[{kname}] b={r['batch']} r{rnd} steps={st or m['steps']}: "
                        f"{r.get('throughput_per_s')} /s failed={r.get('failed')}")
            reap_all("after diffusers")
            (self.out / "versions").mkdir(exist_ok=True)
            (self.out / "versions" / "diffusers.json").write_text(json.dumps(self.versions("diffusers")))
            if attempt == 1:
                done = self.done_cells(tag)
                if all(("diffusers", kname, c, rnd, st or m["steps"]) in done for c, st in plan):
                    return True
                self.decision("retry", tag=tag, arm="diffusers", knob=kname, round=rnd)
        logs = sorted((self.out / "cells" / tag / "diffusers").glob("*.log"), key=lambda x: x.stat().st_mtime)
        failure(self.key, f"diffusers[{kname}]", tag, f"r{rnd}", tail(logs[-1]) if logs else "no log",
                "recorded as failed after one retry; queue moves on")
        return False

    def tune_concs(self):
        if self.m["kind"] == "image":
            return (1, 8), {1: 8, 8: 24}
        return (1, 4), {1: 2, 4: 4}

    def phase_tune(self, engines):
        concs, nums = self.tune_concs()
        for eng in engines:
            for kname in self.knob_table(eng):
                if self.marker(f"tune_{eng}_{kname}").exists():
                    continue
                if eng == "diffusers":
                    self.diffusers_pass(0, f"tune_{kname}", kname, concs, nums, long=False)
                else:
                    self.served_pass(eng, 0, f"tune_{kname}", kname, concs, nums, long=False,
                                     gate_tag=f"{eng}__{kname}")
                self.marker(f"tune_{eng}_{kname}").touch()
        res = {}
        rows = self.rows()
        for eng in engines:
            for kname in self.knob_table(eng):
                rs = {}
                for r in rows:  # latest row per conc wins (a retry supersedes)
                    if r.get("tag") == f"tune_{kname}" and r.get("arm") == eng and "conc" in r:
                        rs[r["conc"]] = r
                ok = len(rs) == len(concs) and all(r.get("failed") == 0 and r.get("throughput_per_s")
                                                  for r in rs.values())
                score = math.exp(statistics.fmean(math.log(r["throughput_per_s"]) for r in rs.values())) \
                    if ok else None
                res.setdefault(eng, {})[kname] = {"score": score, "per_conc": {c: r.get("throughput_per_s")
                                                                                for c, r in rs.items()},
                                                  "lat_b1": rs.get(1, {}).get("lat_mean")}
        (self.out / "tune.json").write_text(json.dumps(res, indent=1))
        # diffusers' tuned config is chosen on speed alone (it is the reference implementation; its
        # compile drift is part of the yardstick), so it is fixed now, before ref_c is collected
        d = res.get("diffusers", {})
        cands = sorted(((v["score"], k) for k, v in d.items() if v["score"]), reverse=True)
        if cands:
            tuned = self.tuned()
            tuned["diffusers"] = {"config": cands[0][1], "candidates": cands}
            (self.out / "tuned.json").write_text(json.dumps(tuned, indent=1))
            self.decision("tuned_config", engine="diffusers", config=cands[0][1], candidates=cands,
                          rule="highest geometric-mean throughput over the tune points")

    def phase_select(self):
        """Rule: a competitor runs its knob set with the highest tune score (geometric mean of
        throughput at the tune points) among the sets whose gate verdict is PASS (incl. distribution-
        level). If none passes, its fastest set runs and is labelled gate-FAIL."""
        tune = json.loads((self.out / "tune.json").read_text())
        try:
            verdict = json.loads((self.out / "gate" / "gate.json").read_text())["verdict"]
        except (OSError, ValueError, KeyError):
            verdict = {}
        tuned = self.tuned()
        for eng, sets in tune.items():
            if eng == "diffusers":
                continue
            bad = self.knob_failed()
            scored = sorted(((v["score"], k) for k, v in sets.items() if v.get("score") and f"{eng}__{k}" not in bad),
                            reverse=True)
            passing = [(s, k) for s, k in scored if verdict.get(f"{eng}__{k}", {}).get("verdict", "").startswith("PASS")]
            pick = (passing or scored or [(None, next(iter(self.knob_table(eng))))])[0]
            tuned[eng] = {"config": pick[1], "score": pick[0], "gate_pass": bool(passing),
                          "candidates": [(s, k, verdict.get(f"{eng}__{k}", {}).get("verdict")) for s, k in scored]}
            self.decision("tuned_config", engine=eng, config=pick[1], gate_pass=bool(passing),
                          candidates=tuned[eng]["candidates"],
                          rule="best tune score among gate-passing knob sets")
        (self.out / "tuned.json").write_text(json.dumps(tuned, indent=1))

    def phase_rounds(self, arms, rounds, first=1):
        for rnd in range(first, first + rounds):
            k = (rnd - 1) % len(arms)
            order = arms[k:] + arms[:k]
            log(f"== round {rnd}: {order}")
            for arm in order:
                if arm == "diffusers":
                    self.diffusers_pass(rnd, "main")
                else:
                    self.served_pass(arm, rnd, "main")

    def phase_profile(self, conc, num):
        """mstar-pr under MSTAR_PHASE_TIMING=100 (untimed, alone on the box), rendered offline with
        benchmark.worker_phases (parse_log -> segment -> render)."""
        t = TREES["mstar-pr"]
        d = self.out / "profile"
        d.mkdir(exist_ok=True)
        reap_all("before profile")
        label = f"profile_c{conc}"
        STATUS.update(phase="profile", arm="mstar-pr", cell=f"c={conc}")
        name = self.start("mstar-pr", label, 0, PORT0, [], env_extra={"MSTAR_PHASE_TIMING": "100"})
        if not name:
            return False
        try:
            s = self.client("mstar-pr", PORT0, d / f"client_c{conc}", conc, num, 2)
            log(f"profile client c={conc}: failed={s.get('failed')}")
        finally:
            self.stop(name)
            reap_all("after profile")
        code = ("import sys; from benchmark.worker_phases.phases import parse_log, segment, render; "
                "print(render(segment(parse_log(open(sys.argv[1]).read()))))")
        r = subprocess.run([f"{t}/.venv/bin/python", "-c", code, str(self.out / "servers" / f"{label}.log")],
                           cwd=t, capture_output=True, text=True, timeout=600)
        (d / f"phases_c{conc}.txt").write_text(r.stdout + r.stderr)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "collect", "gate", "tune", "select", "rounds", "profile"])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--knob", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arms", default="mstar-pr,mstar-main,vllmomni,sglang")
    ap.add_argument("--refs", default="ref_a,ref_b,ref_k,ref_seed")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--engines", default="vllmomni,sglang,diffusers")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--first-round", type=int, default=1)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--num", type=int, default=48)
    a = ap.parse_args()
    b = Bench(a.model, a.out)
    arms = [x for x in a.arms.split(",") if x]
    if a.phase == "smoke":
        wait_gpus_free([a.gpu])
        b.phase_smoke(a.arm, a.gpu, a.knob)
    elif a.phase == "collect":
        b.phase_collect(arms, [x for x in a.refs.split(",") if x], [int(g) for g in a.gpus.split(",")])
    elif a.phase == "gate":
        b.phase_gate(arms)
    elif a.phase == "tune":
        b.phase_tune([x for x in a.engines.split(",") if x])
    elif a.phase == "select":
        b.phase_select()
    elif a.phase == "rounds":
        b.phase_rounds(arms, a.rounds, a.first_round)
    elif a.phase == "profile":
        b.phase_profile(a.conc, a.num)


if __name__ == "__main__":
    main()
