"""Overnight paper-quality benchmark (2026-10-21): M* (PR branch, upstream main) vs vLLM vs
SGLang, one model at a time, one server on the box at a time.

Phases (each its own subcommand, resumable from stage markers under <out>/stages/):
  data         build sharegpt.jsonl and random_1024_256.jsonl with the model's tokenizer
  correct      per arm: start, smoke, greedy parity answers (parity.py collect) and GSM8K;
               mstar-pr twice (pr_a, pr_b: the same-code control). Then the transformers
               oracle (no server up), gate.py and gallery.py.
  tune         competitors' knob sweep (TUNING) on random 1024/256 at c=1, 32, 128;
               best by geometric-mean output tok/s -> tuned.json
  main         ROUNDS x arms; each round restarts every server (fresh process) and rotates
               the order. Each server: smoke, 5 warmup waves before its first cell, 1 before
               the rest. Workloads sharegpt and random_1024_256, c = CONCS.
  xcheck       one point (random 1024/256, c=32) per arm with our client, `vllm bench serve`
               and `sglang.bench_serving`, on the same server
  profile      mstar-pr with MSTAR_PHASE_TIMING=100 at the given points, rendered with
               benchmark/worker_phases
  summarize    REPORT-ready markdown tables

Pass criteria for the correctness gate are fixed here, before the run: gate.py defaults
(mean argmax agreement >= 0.95 and >= worst peer - 0.02, confident-position agreement 1.0,
first-disagreement gap <= max(1.0, worst peer gap), served-token rank < 5, no empty
completion); GSM8K: mstar-pr >= min(vLLM, SGLang) - 0.03 (n=250, about 1 binomial sd at p=0.5).

  python bench.py <phase> --model q35_0.8b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path("/root/work")
RESULTS = Path("/workspace/results/bench")
ENGINES = WORK / "engines"
SHAREGPT = next(iter((WORK / "hf/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots").glob(
    "*/ShareGPT_V3_unfiltered_cleaned_split.json")), None) if (WORK / "hf/hub").exists() else None
PORT = 8300
CLIENT_PY = WORK / "clientenv/bin/python"   # one client env (not any arm's venv) for every arm

TREES = {"mstar-pr": WORK / "mstar", "mstar-main": WORK / "mstar-main", "mstar-pr-cp": WORK / "mstar-cp"}
# "<arm>-rf": the same tree served through its Rust HTTP frontend (`--rust-frontend`)
for _a in list(TREES):
    TREES[_a + "-rf"] = TREES[_a]
ARM_ORDER = ("mstar-pr-rf", "mstar-pr", "mstar-pr-cp-rf", "mstar-pr-cp", "mstar-main-rf", "mstar-main", "vllm", "sglang",
             "vllm-omni", "sglang-omni")
RUST_BIN = "rust/server/target/release/mstar-server"

MODELS = {
    "q35_0.8b": dict(hf="Qwen/Qwen3.5-0.8B", mstar="qwen3_5_0.8b", config="configs/qwen3_5_0.8b.yaml", gpus="0"),
    "q35_9b": dict(hf="Qwen/Qwen3.5-9B", mstar="qwen3_5_9b", config="configs/qwen3_5_9b.yaml", gpus="0"),
    "q35_2b": dict(hf="Qwen/Qwen3.5-2B", mstar="qwen3_5_2b", config="configs/qwen3_5_2b.yaml", gpus="0"),
    "q35_4b": dict(hf="Qwen/Qwen3.5-4B", mstar="qwen3_5_4b", config="configs/qwen3_5_4b.yaml", gpus="0"),
    "q35_9b_tp2": dict(hf="Qwen/Qwen3.5-9B", mstar="qwen3_5_9b", config="configs/qwen3_5_9b_tp2.yaml", gpus="0,1"),
    "q35_27b_tp4": dict(hf="Qwen/Qwen3.5-27B", mstar="qwen3_5_27b", config="configs/qwen3_5_27b_tp4.yaml",
                        gpus="0,1,2,3"),
    # image/video -> text: the same checkpoints and M* configs, vision workloads (vision_data.py)
    "q35_0.8b_vl": dict(hf="Qwen/Qwen3.5-0.8B", mstar="qwen3_5_0.8b", config="configs/qwen3_5_0.8b.yaml", gpus="0",
                        kind="vision"),
    "q35_9b_vl": dict(hf="Qwen/Qwen3.5-9B", mstar="qwen3_5_9b", config="configs/qwen3_5_9b.yaml", gpus="0",
                      kind="vision"),
    # Qwen3-Omni: M*'s shipped default 2-GPU config; vllm-omni's and sglang-omni's default
    # speech deploy configs also use 2 GPUs (thinker on GPU 0, talker on GPU 1)
    "qwen3_omni": dict(hf="Qwen/Qwen3-Omni-30B-A3B-Instruct", mstar="qwen3_omni", config="configs/qwen3omni_2gpu.yaml",
                       gpus="0,1", kind="omni"),
}
DATA_SHARED = {"vision": "vision", "omni": "omni"}
# client options per workload (rows carry their own media and max_tokens)
WL_CLIENT = {
    "omni_t2t": ["--extra-body", '{"modalities": ["text"]}', "--count-actual", "--no-ignore-eos"],
    "omni_a2t": ["--extra-body", '{"modalities": ["text"]}', "--count-actual", "--no-ignore-eos", "--save-text"],
}
RUNNER_SYSTEM = {"vllm-omni": "vllm_omni", "sglang-omni": "sglang_omni"}   # benchmark.runner adapters
# Matched on every arm: mstar's max_seq_len 32768 and gdn_state.max_slots 129 (128 + sink).
MAX_LEN = 32768
MAX_SEQS = 128
CONCS = (1, 2, 4, 8, 16, 32, 64, 128)
WORKLOADS = ("sharegpt", "random_1024_256")

# Knobs per competitor for `tune`: (name, extra args). Every knob runs on top of the matched
# base flags. A knob the version rejects fails its start and is recorded, not fatal.
TUNING = {
    "vllm": [
        ("default", []),
        ("perf-throughput", ["--performance-mode", "throughput"]),
        ("perf-interactivity", ["--performance-mode", "interactivity"]),
        ("lm-only", ["--language-model-only"]),
        ("mnbt-16k", ["--max-num-batched-tokens", "16384"]),
        ("attn-flashinfer", ["--attention-backend", "FLASHINFER"]),
        # second stage: the best single knob combined, and async scheduling named explicitly
        ("lm-only+interactivity", ["--language-model-only", "--performance-mode", "interactivity"]),
        ("lm-only+async", ["--language-model-only", "--async-scheduling"]),
    ],
    "sglang": [
        ("default", []),
        ("attn-fa3", ["--attention-backend", "fa3"]),
        ("attn-flashinfer", ["--attention-backend", "flashinfer"]),
        ("linattn-flashinfer", ["--linear-attn-backend", "flashinfer"]),
        ("chunk-16k", ["--chunked-prefill-size", "16384"]),
        ("torch-compile", ["--enable-torch-compile", "--torch-compile-max-bs", "128"]),
        ("lang-only", ["--language-only"]),
        # torch.compile's capture OOMs at --mem-fraction-static 0.85; give it room
        ("torch-compile-mf0.7", ["--enable-torch-compile", "--torch-compile-max-bs", "128",
                                 "--mem-fraction-static", "0.7"]),
        ("fa3+lang-only", ["--attention-backend", "fa3", "--language-only"]),
    ],
}
TUNE_POINTS = ((1, 8), (32, 96), (128, 256))   # (concurrency, requests)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


BAD_LOG = ("Traceback", "illegal memory access", "CUDA error")
WATCH_PERIOD = 30          # s between liveness checks while a cell runs
GPU_IDLE_LIMIT = 300       # s of 0% utilisation on every model GPU that aborts a timed cell


def server_pids() -> list[int]:
    """The benchmark server (server.sh's recorded pid and listener) and every descendant."""
    st = Path(os.environ.get("SERVER_STATE_DIR") or f"/tmp/servers_{os.environ.get('USER', '')}")
    roots = []
    for f in ("bench.pid", "bench.listener"):
        try:
            roots.append(int((st / f).read_text().strip()))
        except (OSError, ValueError):
            pass
    out = subprocess.run(["ps", "-e", "-o", "pid=,ppid=,etimes=,args="], capture_output=True, text=True).stdout
    parent, age, helper = {}, {}, set()
    for line in out.splitlines():
        parts = line.split(None, 3)
        try:
            pid, ppid, et = (int(x) for x in parts[:3])
        except ValueError:
            continue
        parent[pid], age[pid] = ppid, et
        # torch Inductor's compile-worker pool is long-lived but recycles itself on a lazy compile;
        # its exit is not the server dying.
        if len(parts) > 3 and ("compile_worker" in parts[3] or "_inductor" in parts[3]):
            helper.add(pid)
    keep = set(r for r in roots if r in parent)
    for pid in parent:
        q = pid
        while q in parent and q > 1:
            if q in keep:
                keep.add(pid)
                break
            q = parent[q]
    return sorted(p for p in keep if age.get(p, 0) >= 60 and p not in helper)   # long-lived workers only


def run_watched(cmd, timeout, gpus, server_log, **kw):
    """subprocess.run(capture_output=True, text=True) for a timed client, with a liveness
    watchdog: abort at once if the server's /health stops answering (twice in a row), one of
    its long-lived processes exits, its log gains a Traceback / illegal memory access / CUDA
    error, or every model GPU sits at 0% for GPU_IDLE_LIMIT s. Returns (proc, reason)."""
    import tempfile
    base_pids = set(server_pids())
    try:
        log_off = Path(server_log).stat().st_size if server_log else 0
    except OSError:
        log_off = 0
    out_f, err_f = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True, **kw)
    t0, idle_since, health_miss, reason = time.time(), None, 0, None
    while True:
        try:
            proc.wait(timeout=WATCH_PERIOD)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        if now - t0 > timeout:
            reason = f"cell timed out after {timeout}s"
        if not reason:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
                    health_miss = 0 if r.status == 200 else health_miss + 1
            except Exception:  # noqa: BLE001
                health_miss += 1
            if health_miss >= 2:
                reason = "watchdog: server /health stopped answering"
        if not reason and base_pids:
            gone = base_pids - set(server_pids())
            if gone:
                reason = f"watchdog: server process(es) exited: {sorted(gone)[:5]}"
        if not reason and server_log:
            try:
                with open(server_log, errors="replace") as f:
                    f.seek(log_off)
                    new = f.read()
                    log_off = f.tell()
                hit = next((b for b in BAD_LOG if b in new), None)
                if hit:
                    line = next(ln for ln in new.splitlines() if hit in ln)
                    reason = f"watchdog: server log: {line.strip()[:200]}"
            except OSError:
                pass
        if not reason and gpus:
            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                                "-i", gpus], capture_output=True, text=True).stdout.split()
            if o and all(x.strip() == "0" for x in o):
                idle_since = idle_since or now
                if now - idle_since > GPU_IDLE_LIMIT:
                    reason = f"watchdog: GPUs {gpus} at 0% for {GPU_IDLE_LIMIT}s"
            else:
                idle_since = None
        if reason:
            proc.kill()
            proc.wait()
            break
    out_f.seek(0)
    err_f.seek(0)
    res = subprocess.CompletedProcess(cmd, proc.returncode, out_f.read(), err_f.read() + (f"\n{reason}" if reason else ""))
    return res, reason


def loadavg():
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return None


class Bench:
    def __init__(self, model: str, out: Path | None = None):
        self.key = model
        self.m = MODELS[model]
        self.out = out or RESULTS / model
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "stages").mkdir(exist_ok=True)
        self.tp = len(self.m["gpus"].split(","))
        self.kind = self.m.get("kind", "text")
        self.data = RESULTS / "_data" / DATA_SHARED[self.kind] if self.kind in DATA_SHARED else self.out / "data"

    # ---- servers ---------------------------------------------------------------------
    def tuned(self) -> dict:
        p = self.out / "tuned.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def server_command(self, arm: str, extra: list[str] | None = None, env_extra: dict | None = None):
        env = dict(os.environ)
        env["HF_HOME"] = str(WORK / "hf")
        env["HF_HUB_OFFLINE"] = "1"
        extra = list(extra or [])
        if arm in TREES:
            tree = TREES[arm]
            env["PATH"] = f"{tree}/.venv/bin:{env['PATH']}"
            env.update(MSTAR_SHM_ARENA="1", MSTAR_REQUIRE_ACCELERATOR_GRAPHS="1")
            cmd = [f"{tree}/.venv/bin/python", "-m", "mstar.cli.main", "serve", self.m["mstar"],
                   "--config", str(tree / self.m["config"]), "--gpus", self.m["gpus"], "--port", str(PORT)]
            if arm.endswith("-rf"):
                cmd += ["--rust-frontend", "--rust-frontend-bin", str(tree / RUST_BIN)]
            cwd = tree
        elif arm == "vllm":
            v = ENGINES / "vllm"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env["CUDA_VISIBLE_DEVICES"] = self.m["gpus"]
            cmd = [f"{v}/bin/vllm", "serve", self.m["hf"], "--tensor-parallel-size", str(self.tp),
                   "--max-model-len", str(MAX_LEN), "--max-num-seqs", str(MAX_SEQS),
                   "--no-enable-prefix-caching", "--gpu-memory-utilization", "0.9",
                   "--host", "127.0.0.1", "--port", str(PORT)]
            if self.kind == "vision":   # no multimodal processor cache (a hit would skip preprocessing)
                cmd += ["--mm-processor-cache-gb", "0", "--limit-mm-per-prompt", '{"image": 1, "video": 1}']
            cwd = ENGINES
        elif arm == "sglang":
            v = ENGINES / "sglang"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env["CUDA_VISIBLE_DEVICES"] = self.m["gpus"]
            cmd = [f"{v}/bin/python", "-m", "sglang.launch_server", "--model-path", self.m["hf"],
                   "--tp", str(self.tp), "--context-length", str(MAX_LEN),
                   "--max-running-requests", str(MAX_SEQS), "--disable-radix-cache",
                   "--mamba-ssm-dtype", "float32", "--mem-fraction-static", "0.85",
                   "--host", "127.0.0.1", "--port", str(PORT)]
            if self.kind == "vision":   # no vision-embedding cache (a hit would skip the encoder)
                env["SGLANG_VLM_CACHE_SIZE_MB"] = "0"
            cwd = ENGINES
        elif arm == "vllm-omni":
            v = ENGINES / "vllm-omni"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env["CUDA_VISIBLE_DEVICES"] = self.m["gpus"]
            # its recommended default: vllm_omni/deploy/qwen3_omni_moe.yaml (thinker GPU 0, talker +
            # code2wav GPU 1, prefix caching off, max_num_seqs 64 per stage)
            cmd = [f"{v}/bin/vllm", "serve", self.m["hf"], "--omni", "--host", "127.0.0.1", "--port", str(PORT)]
            cwd = ENGINES
        elif arm == "sglang-omni":
            v = ENGINES / "sglang-omni"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env["CUDA_VISIBLE_DEVICES"] = self.m["gpus"]
            # its default speech pipeline: thinker (+ encoders, code2wav) GPU 0, talker GPU 1.
            # The talker's first start (compile + graph capture) runs past the default 600 s stall
            # timeout; the error message names this knob.
            env["SGLANG_OMNI_STARTUP_TIMEOUT"] = "3600"
            cmd = [f"{v}/bin/python", "-m", "sglang_omni.cli", "serve", "--model-path", self.m["hf"],
                   "--host", "127.0.0.1", "--port", str(PORT)]
            cwd = ENGINES
        else:
            raise ValueError(arm)
        if env_extra:
            env.update(env_extra)
        return cmd + extra, env, cwd

    def arm_args(self, arm: str) -> tuple[str, list[str]]:
        """The knob set an arm runs in timed passes: competitors' tuned best."""
        t = self.tuned().get(arm)
        if t:
            return t["config"], t["args"]
        return "default", []

    @staticmethod
    def gpu_mem():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return [int(x) for x in out.split()]

    def wait_box_free(self, timeout=300):
        """Only one server on the whole box: every GPU must be free before a start."""
        t0 = time.time()
        while max(used := self.gpu_mem()) > 2048:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"GPUs still in use {used} MiB; refusing to share")
            time.sleep(5)

    def start(self, arm: str, label: str, extra=None, env_extra=None, timeout: int | None = None) -> bool:
        self.wait_box_free()
        cmd, env, cwd = self.server_command(arm, extra, env_extra)
        srv_log = self.out / "servers" / f"{label}.log"
        self.cur_log = srv_log
        srv_log.parent.mkdir(exist_ok=True)
        (self.out / "servers" / f"{label}.cmd").write_text(shlex.join(cmd) + "\n" + json.dumps(
            {k: env[k] for k in env if k.startswith(("MSTAR_", "VLLM_", "SGLANG_", "CUDA_VISIBLE"))}) + "\n")
        log(f"start {label}: {shlex.join(cmd)}")
        t0 = time.time()
        timeout = timeout or (3600 if arm in TREES else 1800)
        r = subprocess.run([str(HERE / "server.sh"), "start", "bench", str(PORT), "--timeout", str(timeout),
                            "--log", str(srv_log), "--", *cmd], env=env, cwd=cwd)
        if r.returncode != 0:
            log(f"!! {label}: start failed (see {srv_log})")
            self.stop()
            return False
        with (self.out / "startup.jsonl").open("a") as f:
            f.write(json.dumps({"label": label, "arm": arm, "seconds": round(time.time() - t0, 1)}) + "\n")
        return True

    def stop(self):
        subprocess.run([str(HERE / "server.sh"), "stop", "bench"], capture_output=True)

    def smoke(self) -> str:
        body = {"model": self.m["hf"], "max_tokens": 24, "temperature": 0.0, "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "Say hello in five words."}],
                **({"modalities": ["text"]} if self.kind == "omni" else {})}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            msg = json.load(r)["choices"][0]["message"]
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or msg.get("reasoning") or "")
        if not text.strip():
            raise RuntimeError("smoke returned no text")
        return text[:80]

    def versions(self, arm: str, label: str):
        if arm in TREES:
            tree = TREES[arm]
            code = ("import mstar, mstar_rust, torch, flashinfer, transformers; "
                    "print(mstar.__file__, torch.__version__, flashinfer.__version__, transformers.__version__)")
            info = {"imports": subprocess.run([f"{tree}/.venv/bin/python", "-c", code], capture_output=True,
                                              text=True, cwd=tree).stdout.strip(),
                    "commit": (tree / "COMMIT.txt").read_text().strip() if (tree / "COMMIT.txt").exists() else None}
        else:
            fr = subprocess.run(["uv", "pip", "freeze", "--python", str(ENGINES / arm / "bin/python")],
                                capture_output=True, text=True, env={**os.environ,
                                "PATH": f"/root/.local/bin:{os.environ['PATH']}"}).stdout
            keep = ("vllm", "sglang", "torch=", "flashinfer", "transformers", "flash-attn", "triton")
            info = {"packages": [x for x in fr.splitlines() if x.lower().startswith(keep)]}
        with (self.out / "versions.jsonl").open("a") as f:
            f.write(json.dumps({"label": label, "arm": arm, **info}) + "\n")

    # ---- cells -----------------------------------------------------------------------
    def cell(self, tag: str, arm: str, config: str, wl: str, conc: int, rnd: int, warm: int,
             num: int | None = None, timeout: float = 1800) -> dict:
        d = self.out / "cells" / tag / wl / arm / f"c{conc}_r{rnd}"
        d.mkdir(parents=True, exist_ok=True)
        n = num or max(16, 8 * conc)
        cmd = [str(CLIENT_PY), "-I", str(HERE / "client.py"), "--url", f"http://127.0.0.1:{PORT}",
               "--model-id", self.m["hf"], "--data", str(self.data / f"{wl}.jsonl"), "--conc", str(conc),
               "--num", str(n), "--warmup-waves", str(warm), "--warmup-offset", "1100",
               "--tokenizer", self.m["hf"], "--out", str(d)] + WL_CLIENT.get(wl, [])
        mon = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                                "-lms", "500", "-i", self.m["gpus"]], stdout=subprocess.PIPE, text=True)
        la0, t0 = loadavg(), time.time()
        p, reason = run_watched(cmd, timeout, self.m["gpus"], getattr(self, "cur_log", None),
                                env={**os.environ, "HF_HOME": str(WORK / "hf"), "HF_HUB_OFFLINE": "1"})
        secs, la1 = time.time() - t0, loadavg()
        mon.terminate()
        util_raw = mon.communicate()[0].split()
        (d / "client.log").write_text(p.stdout + p.stderr)
        row = {"tag": tag, "arm": arm, "config": config, "workload": wl, "conc": conc, "round": rnd,
               "warmup_waves": warm, "num": n, "seconds": round(secs, 1), "load_before": la0, "load_after": la1,
               "started": t0, "notes": []}
        try:
            util = [int(x) for x in util_raw if x.strip().isdigit()]
            row["gpu_util_mean"] = round(statistics.mean(util), 1) if util else None
        except ValueError:
            row["gpu_util_mean"] = None
        try:
            s = json.loads((d / "results.json").read_text())["summary"]
            row.update({k: s[k] for k in ("completed", "failed", "output_tok_s", "total_tok_s", "ttft_p50",
                                          "ttft_p99", "tpot_p50", "tpot_p99", "itl_p50", "itl_p99", "e2e_p50",
                                          "mean_output_len", "mean_input_len", "chunks_per_token", "retok_ratio",
                                          "usage_reported", "wall_s", "errors")})
        except (OSError, ValueError, KeyError) as e:
            row["notes"].append(f"no results.json: {e!r}")
            row["failed"] = None
        if reason:
            row["notes"].append(reason)
        elif p.returncode != 0:
            row["notes"].append(f"client exit {p.returncode}: {p.stderr[-300:]}")
        with (self.out / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        tps = row.get("output_tok_s")
        log(f"   {tag} {arm}[{config}] {wl} c={conc} r{rnd}: {tps and round(tps)} tok/s "
            f"ttft50={row.get('ttft_p50') and round(row['ttft_p50'] * 1e3, 1)}ms "
            f"tpot50={row.get('tpot_p50') and round(row['tpot_p50'] * 1e3, 2)}ms failed={row.get('failed')} "
            f"util={row.get('gpu_util_mean')} {secs:.0f}s {' '.join(row['notes'])}")
        return row

    def cell_runner(self, tag: str, arm: str, config: str, conc: int, rnd: int, warm: int,
                    num: int | None = None, timeout: float = 1800, out_dir: Path | None = None) -> dict:
        """Qwen3-Omni text->speech through the repo's own benchmark.runner (a copy of main's
        benchmark/ in omni_client/), which has adapters for M*, vllm-omni and sglang-omni
        audio streams; every request's text and WAV are saved for the Whisper WER check."""
        import re
        wl = "omni_t2s"
        d = out_dir or self.out / "cells" / tag / wl / arm / f"c{conc}_r{rnd}"
        d.mkdir(parents=True, exist_ok=True)
        n = num or max(16, 8 * conc)
        system = "ours_openai" if arm in TREES else RUNNER_SYSTEM[arm]
        cmd = [str(WORK / "mstar-main/.venv/bin/python"), "-m", "benchmark.runner", "--url", f"http://127.0.0.1:{PORT}",
               "--model", "qwen3omni", "--inference-system", system, "--profiling-type", "closed_loop",
               "--request-type", "text_to_speech", "--max-concurrency", str(conc), "--num-requests", str(n),
               "--num-warmup", str(warm), "--output-dir", str(d), "--local-cache", str(RESULTS / "_data/omni/cache")]
        row = {"tag": tag, "arm": arm, "config": config, "workload": wl, "conc": conc, "round": rnd,
               "warmup_waves": warm, "num": n, "load_before": loadavg(), "started": time.time(), "notes": []}
        t0 = time.time()
        p, reason = run_watched(cmd, timeout, self.m["gpus"], getattr(self, "cur_log", None),
                                cwd=str(HERE / "omni_client"), env={**os.environ, "HF_HOME": str(WORK / "hf")})
        out = p.stdout + p.stderr
        if reason:
            row["notes"].append(reason)
        elif p.returncode != 0:
            row["notes"].append(f"runner exit {p.returncode}: {p.stderr[-300:]}")
        row["seconds"], row["load_after"] = round(time.time() - t0, 1), loadavg()
        (d / "bench.log").write_text(out)
        row["log"] = str(d / "bench.log")

        def stat(label, k):
            line = re.search(r"^\s*" + r"\s+".join(map(re.escape, label.split())) + r"[^\n]*", out, re.M)
            if not line:
                return None
            m = re.search(r"\b" + re.escape(k) + r"\s*=\s*([0-9]*\.?[0-9]+)", line.group(0))
            return float(m.group(1)) if m else None
        try:
            res = json.loads((d / "results.json").read_text())
            row["completed"], row["failed"] = res.get("completed"), res.get("failed")
        except (OSError, ValueError) as e:
            row["notes"].append(f"no results.json: {e!r}")
            row["failed"] = None
        m = re.search(r"Throughput:\s*([0-9.]+)\s*text tok/s", out)
        row["output_tok_s"] = float(m.group(1)) if m else None
        m = re.search(r"Throughput:\s*([0-9.]+)\s*audio sec/s", out)
        row["audio_s_s"] = float(m.group(1)) if m else None
        row["ttft_p50"], row["ttft_p99"] = stat("TTFT (text)", "p50"), stat("TTFT (text)", "p99")
        row["ttfa_p50"], row["ttfa_p99"] = stat("TTFT (audio)", "p50"), stat("TTFT (audio)", "p99")
        row["rtf_p50"], row["rtf_p99"] = stat("RTF", "p50"), stat("RTF", "p99")
        m = re.search(r"Audio dur\s*:\s*mean=([0-9.]+)s", out)
        row["audio_mean_s"] = float(m.group(1)) if m else None
        m = re.search(r"Text tokens:\s*[0-9]+\s*total\s*\(([0-9.]+)\s*avg/req\)", out)
        row["mean_output_len"] = float(m.group(1)) if m else None
        if not row.get("audio_s_s") and row.get("failed") == 0:
            row["notes"].append("no audio produced")
        with (self.out / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"   {tag} {arm} {wl} c={conc} r{rnd}: audio {row.get('audio_s_s')} s/s, text {row.get('output_tok_s')} tok/s "
            f"ttfa50={row.get('ttfa_p50')}s rtf50={row.get('rtf_p50')} failed={row.get('failed')} {' '.join(row['notes'])}")
        return row

    def marker(self, name: str) -> Path:
        return self.out / "stages" / f"{name}.done"

    # ---- phases ----------------------------------------------------------------------
    def phase_data(self):
        cmd = [str(ENGINES / "vllm/bin/python"), "-I", str(HERE / "make_data.py"), "--tokenizer", self.m["hf"],
               "--sharegpt", str(SHAREGPT), "--out", str(self.data)]
        subprocess.run(cmd, check=True, env={**os.environ, "HF_HOME": str(WORK / "hf"), "HF_HUB_OFFLINE": "1"})

    def phase_correct(self, arms):
        pdir = self.out / "parity"
        pdir.mkdir(exist_ok=True)
        prompts = HERE / "prompts_text.jsonl"
        for arm in arms:
            reps = ["a", "b"] if arm == "mstar-pr" else ["a"]
            for rep in reps:
                tag = f"{arm}_{rep}".replace("-", "_")
                if self.marker(f"correct_{tag}").exists():
                    continue
                config, args = self.arm_args(arm)
                if not self.start(arm, f"correct_{tag}", args):
                    continue
                try:
                    log(f"== correct {tag} smoke: {self.smoke()!r}")
                    self.versions(arm, f"correct_{tag}")
                    subprocess.run([str(CLIENT_PY), "-I", str(HERE / "parity.py"), "collect", "--port", str(PORT),
                                    "--tag", tag, "--model-id", self.m["hf"], "--prompts", str(prompts),
                                    "--out-dir", str(pdir)], check=True)
                    if rep == "a":
                        subprocess.run([str(CLIENT_PY), "-I", str(HERE / "gsm8k.py"), "--url", f"http://127.0.0.1:{PORT}",
                                        "--model-id", self.m["hf"], "--n", "250", "--conc", "32",
                                        "--out", str(self.out / f"gsm8k_{tag}.json")], check=True,
                                       env={**os.environ, "HF_HOME": str(WORK / "hf")})
                    self.marker(f"correct_{tag}").touch()
                except Exception as e:  # noqa: BLE001
                    log(f"!! correct {tag}: {e!r}")
                finally:
                    self.stop()

    def phase_oracle(self, prompts: Path | None = None, pdir: Path | None = None, gate_name: str = "gate.txt",
                     peers_from=("vllm_a", "sglang_a")):
        prompts = prompts or HERE / "prompts_text.jsonl"
        pdir = pdir or self.out / "parity"
        self.wait_box_free()
        dev = "auto" if self.tp > 1 else "cuda:0"
        env = {**os.environ, "HF_HOME": str(WORK / "hf"), "HF_HUB_OFFLINE": "1",
               "CUDA_VISIBLE_DEVICES": self.m["gpus"], "NVIDIA_TF32_OVERRIDE": "0"}
        with (self.out / f"oracle_{gate_name.replace('.txt', '')}.log").open("w") as f:
            subprocess.run([str(ENGINES / "vllm/bin/python"), "-I", str(HERE / "parity.py"), "oracle", "--model-id",
                            self.m["hf"], "--prompts", str(prompts), "--out-dir", str(pdir),
                            "--device", dev], env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
        peers = ",".join(t for t in peers_from if (pdir / f"{t}.json").exists())
        tags = [t for t in ("mstar_pr_a", "mstar_pr_rf_a", "mstar_pr_cp_a", "mstar_pr_cp_rf_a", "mstar_main_a",
                            "mstar_main_rf_a", "vllm_mtp", "sglang_mtp") if (pdir / f"{t}.json").exists()]
        with (self.out / gate_name).open("w") as f:
            for tag, ctl, pr in ((t, "mstar_pr_b", peers) for t in tags):
                f.write(f"#### gate {tag}\n")
                f.flush()
                subprocess.run([sys.executable, "-I", str(HERE / "gate.py"), "--parity", str(pdir), "--tag", tag,
                                "--control", ctl, "--peers", pr], stdout=f, stderr=subprocess.STDOUT)
        subprocess.run([sys.executable, "-I", str(HERE / "gallery.py"), "--prompts", str(prompts),
                        "--parity", str(pdir), "--out", str(self.out / ("gallery" if gate_name == "gate.txt"
                                                                        else gate_name.replace(".txt", "_gallery"))),
                        "--title", f"{self.key}: mstar-pr / mstar-main / vLLM / SGLang / transformers"])
        print((self.out / gate_name).read_text())

    def phase_tune(self, engines, knobs=None):
        for eng in engines:
            for name, args in TUNING[eng]:
                if knobs and name not in knobs:
                    continue
                label = f"tune_{eng}_{name}"
                if self.marker(label).exists():
                    continue
                if not self.start(eng, label, args):
                    with (self.out / "tune_failed.jsonl").open("a") as f:
                        f.write(json.dumps({"engine": eng, "knob": name, "args": args}) + "\n")
                    self.marker(label).touch()
                    continue
                try:
                    log(f"== {label} smoke: {self.smoke()!r}")
                    first = True
                    for c, n in TUNE_POINTS:
                        self.cell("tune", eng, name, "random_1024_256", c, 1, 3 if first else 1, num=n)
                        first = False
                    self.marker(label).touch()
                except Exception as e:  # noqa: BLE001
                    log(f"!! {label}: {e!r}")
                finally:
                    self.stop()
        self.pick_tuned()

    def phase_fetune(self, tree_arm="mstar-pr"):
        """M*'s own knob: Python (uvicorn/FastAPI) vs Rust (mstar-server) HTTP frontend, same
        short sweep as the competitors' tuning, plus the correctness answers for the Rust one."""
        pdir = self.out / "parity"
        for arm in (tree_arm, tree_arm + "-rf"):
            label = f"tune_{arm}"
            if self.marker(label).exists():
                continue
            if not self.start(arm, label):
                with (self.out / "tune_failed.jsonl").open("a") as f:
                    f.write(json.dumps({"engine": arm, "knob": "frontend", "args": []}) + "\n")
                self.marker(label).touch()
                continue
            try:
                log(f"== {label} smoke: {self.smoke()!r}")
                self.versions(arm, label)
                if arm.endswith("-rf"):
                    tag = f"{arm}_a".replace("-", "_")
                    subprocess.run([str(CLIENT_PY), "-I", str(HERE / "parity.py"), "collect", "--port", str(PORT),
                                    "--tag", tag, "--model-id", self.m["hf"], "--prompts",
                                    str(HERE / "prompts_text.jsonl"), "--out-dir", str(pdir)], check=True)
                    subprocess.run([str(CLIENT_PY), "-I", str(HERE / "gsm8k.py"), "--url", f"http://127.0.0.1:{PORT}",
                                    "--model-id", self.m["hf"], "--n", "250", "--conc", "32",
                                    "--out", str(self.out / f"gsm8k_{tag}.json")], check=True,
                                   env={**os.environ, "HF_HOME": str(WORK / "hf")})
                first = True
                for c, n in TUNE_POINTS:
                    self.cell("tune", arm, "rust-frontend" if arm.endswith("-rf") else "python-frontend",
                              "random_1024_256", c, 1, 3 if first else 1, num=n)
                    first = False
                for c in (1, 128):
                    self.cell("tune", arm, "rust-frontend" if arm.endswith("-rf") else "python-frontend",
                              "sharegpt", c, 1, 1, num=max(16, 4 * c))
                self.marker(label).touch()
            except Exception as e:  # noqa: BLE001
                log(f"!! {label}: {e!r}")
            finally:
                self.stop()
        rows = [r for r in self.rows() if r["tag"] == "tune" and r["workload"] == "random_1024_256"]
        score = {}
        for arm in (tree_arm, tree_arm + "-rf"):
            ok = [r for r in rows if r["arm"] == arm and r.get("failed") == 0 and r.get("output_tok_s")]
            if len(ok) == len(TUNE_POINTS):
                score[arm] = math.exp(statistics.mean(math.log(r["output_tok_s"]) for r in ok))
        log(f"== frontend tune {tree_arm}: geomean tok/s {score}")
        t = self.tuned()
        t["mstar_frontend_geomean"] = score
        (self.out / "tuned.json").write_text(json.dumps(t, indent=1))

    def pick_tuned(self):
        rows = self.rows()
        best = self.tuned()
        for eng in TUNING:
            scores = {}
            for name, args in TUNING[eng]:
                if not (self.out / "stages" / f"tune_{eng}_{name}.done").exists():
                    continue
                cells = [r for r in rows if r["tag"] == "tune" and r["arm"] == eng and r["config"] == name]
                ok = [r for r in cells if r.get("failed") == 0 and r.get("output_tok_s")]
                if len(ok) == len(TUNE_POINTS):
                    scores[name] = (math.exp(statistics.mean(math.log(r["output_tok_s"]) for r in ok)), args)
            if scores:
                name = max(scores, key=lambda k: scores[k][0])
                best[eng] = {"config": name, "args": scores[name][1],
                             "geomean_tok_s": {k: round(v[0], 1) for k, v in scores.items()}}
                log(f"== tune {eng}: best {name}; {best[eng]['geomean_tok_s']}")
        (self.out / "tuned.json").write_text(json.dumps(best, indent=1))

    def phase_main(self, rounds, arms, first_arms_rotate=True, concs=CONCS, workloads=WORKLOADS):
        for rnd in rounds:
            order = list(arms)
            if first_arms_rotate and len(order) > 1:
                k = (rnd - 1) % len(order)
                order = order[k:] + order[:k]
            for arm in order:
                label = f"main_{arm}_r{rnd}"
                if self.marker(label).exists():
                    continue
                # a round that broke off is redone whole: one round = one server process
                for wl in workloads:
                    for c in concs:
                        d = self.out / "cells" / "main" / wl / arm / f"c{c}_r{rnd}"
                        if d.exists():
                            dest = self.out / "aborted" / f"{label}_{int(time.time())}" / wl / d.name
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(d), str(dest))
                self.drop_rows(lambda r, arm=arm, rnd=rnd: r["tag"] == "main" and r["arm"] == arm and r["round"] == rnd)
                config, args = self.arm_args(arm)
                if not self.start(arm, label, args):
                    continue
                try:
                    log(f"== {label}[{config}] smoke: {self.smoke()!r}")
                    self.versions(arm, label)
                    first = True
                    for wl in workloads:
                        for c in concs:
                            self.cell("main", arm, config, wl, c, rnd, 5 if first else 1)
                            first = False
                    self.marker(label).touch()
                except Exception as e:  # noqa: BLE001
                    log(f"!! {label}: {e!r}")
                finally:
                    self.stop()
                self.summarize(quiet=True)

    def phase_xcheck(self, arms, conc=32, num=160):
        xdir = self.out / "xcheck"
        xdir.mkdir(exist_ok=True)
        for arm in arms:
            label = f"xcheck_{arm}"
            if self.marker(label).exists():
                continue
            config, args = self.arm_args(arm)
            if not self.start(arm, label, args):
                continue
            try:
                log(f"== {label} smoke: {self.smoke()!r}")
                self.cell("xcheck", arm, config, "random_1024_256", conc, 1, 5, num=num)
                env = {**os.environ, "HF_HOME": str(WORK / "hf"), "HF_HUB_OFFLINE": "1"}
                extra = json.dumps({"temperature": 0.0, "enable_thinking": False,
                                    "chat_template_kwargs": {"enable_thinking": False}})
                base = f"http://127.0.0.1:{PORT}"
                for tool, cmd in (
                    ("vllm_bench", [str(ENGINES / "vllm/bin/vllm"), "bench", "serve", "--backend", "openai-chat",
                                    "--base-url", base, "--endpoint", "/v1/chat/completions", "--model", self.m["hf"],
                                    "--tokenizer", self.m["hf"], "--dataset-name", "random", "--random-input-len",
                                    "1024", "--random-output-len", "256", "--random-range-ratio", "0",
                                    "--num-prompts", str(num), "--max-concurrency", str(conc), "--ignore-eos",
                                    "--num-warmups", str(conc), "--percentile-metrics", "ttft,tpot,itl,e2el",
                                    "--metric-percentiles", "50,99", "--extra-body", extra, "--save-result",
                                    "--result-dir", str(xdir), "--result-filename", f"vllm_bench_{arm}.json",
                                    "--seed", "0"]),
                    ("sglang_bench", [str(ENGINES / "sglang/bin/python"), "-m", "sglang.bench_serving", "--backend",
                                      "sglang-oai-chat" if arm != "vllm" else "vllm-chat", "--base-url", base,
                                      "--model", self.m["hf"], "--tokenizer", self.m["hf"],
                                      "--dataset-name", "random", "--random-input-len", "1024",
                                      "--random-output-len", "256", "--random-range-ratio", "1.0",
                                      "--num-prompts", str(num), "--max-concurrency", str(conc),
                                      "--warmup-requests", str(conc), "--extra-request-body", extra,
                                      "--output-file", str(xdir / f"sglang_bench_{arm}.jsonl"), "--seed", "0"]),
                ):
                    try:
                        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=xdir, timeout=1200)
                    except subprocess.TimeoutExpired:
                        p = subprocess.CompletedProcess(cmd, -9, "", f"{tool} timed out after 1200s")
                    (xdir / f"{tool}_{arm}.log").write_text(p.stdout + p.stderr)
                    tail = [ln for ln in (p.stdout + p.stderr).splitlines()
                            if any(k in ln for k in ("Output token throughput", "Median TTFT", "Median TPOT",
                                                     "Successful requests", "Failed", "Error"))]
                    log(f"   {tool} on {arm}: exit {p.returncode}; " + " | ".join(tail[:8]))
                self.marker(label).touch()
            except Exception as e:  # noqa: BLE001
                log(f"!! {label}: {e!r}")
            finally:
                self.stop()

    MTP = {  # each engine's own Qwen3.5 recipe setting
        "vllm": ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 1}'],
        "sglang": ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
                   "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
                   "--mem-fraction-static", "0.75"],  # 0.85 OOMs under spec decoding at c=128
    }

    def phase_mtp(self, engines, rounds, concs=(1, 4, 16), workloads=WORKLOADS):
        """Supplementary: competitors with their recipe's MTP speculative decoding. Same
        quality gate (parity answers + GSM8K on the first round's server)."""
        pdir = self.out / "parity"
        for rnd in rounds:
            for eng in engines:
                label = f"mtp_{eng}_r{rnd}"
                if self.marker(label).exists():
                    continue
                self.drop_rows(lambda r, eng=eng, rnd=rnd: r["tag"] == "mtp" and r["arm"] == eng and r["round"] == rnd)
                config, args = self.arm_args(eng)
                if not self.start(eng, label, args + self.MTP[eng]):
                    continue
                try:
                    log(f"== {label} smoke: {self.smoke()!r}")
                    self.versions(eng, label)
                    if rnd == rounds[0]:
                        tag = f"{eng}_mtp"
                        subprocess.run([str(CLIENT_PY), "-I", str(HERE / "parity.py"), "collect", "--port", str(PORT),
                                        "--tag", tag, "--model-id", self.m["hf"], "--prompts",
                                        str(HERE / "prompts_text.jsonl"), "--out-dir", str(pdir)], check=True)
                        subprocess.run([str(CLIENT_PY), "-I", str(HERE / "gsm8k.py"), "--url", f"http://127.0.0.1:{PORT}",
                                        "--model-id", self.m["hf"], "--n", "250", "--conc", "32",
                                        "--out", str(self.out / f"gsm8k_{tag}.json")], check=True,
                                       env={**os.environ, "HF_HOME": str(WORK / "hf")})
                    first = True
                    for wl in workloads:
                        for c in concs:
                            self.cell("mtp", eng, config + "+mtp", wl, c, rnd, 5 if first else 1)
                            first = False
                    self.marker(label).touch()
                except Exception as e:  # noqa: BLE001
                    log(f"!! {label}: {e!r}")
                finally:
                    self.stop()

    def phase_profile(self, points, arm="mstar-pr"):
        label = f"profile_{arm}"
        if not self.start(arm, label, env_extra={"MSTAR_PHASE_TIMING": "100"}):
            return
        try:
            log(f"== {label} smoke: {self.smoke()!r}")
            first = True
            for wl, c in points:
                self.cell("profile", arm, "phase_timing", wl, c, 1, 3 if first else 1)
                first = False
        finally:
            self.stop()
        code = ("import sys; sys.path.insert(0, sys.argv[2]); "
                "from benchmark.worker_phases.phases import parse_log, render, segment; "
                "print(render(segment(parse_log(open(sys.argv[1]).read()))))")
        tree = TREES[arm]
        p = subprocess.run([f"{tree}/.venv/bin/python", "-c", code, str(self.out / "servers" / f"{label}.log"),
                            str(tree)], capture_output=True, text=True)
        (self.out / f"phases_{arm}.txt").write_text(p.stdout + p.stderr)
        print(p.stdout[-6000:])

    # ---- results ---------------------------------------------------------------------
    def rows(self):
        p = self.out / "results.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def drop_rows(self, pred):
        p = self.out / "results.jsonl"
        if not p.exists():
            return
        keep = [r for r in self.rows() if not pred(r)]
        p.write_text("".join(json.dumps(r) + "\n" for r in keep))

    def pooled(self, wl, arm, conc, rounds, tag="main"):
        """Latency percentiles over every request of every counted round of one point."""
        ttft, tpot = [], []
        for rnd in rounds:
            f = self.out / "cells" / tag / wl / arm / f"c{conc}_r{rnd}" / "results.json"
            try:
                reqs = json.loads(f.read_text())["requests"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            ttft += [r["ttft"] for r in reqs if r.get("ok")]
            tpot += [r["tpot"] for r in reqs if r.get("ok") and r.get("tpot") is not None]
        def pct(xs, q):
            if not xs:
                return None
            xs = sorted(xs)
            k = (len(xs) - 1) * q / 100.0
            f = int(k)
            c = min(f + 1, len(xs) - 1)
            return xs[f] + (xs[c] - xs[f]) * (k - f)
        return {"ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99), "tpot_p50": pct(tpot, 50),
                "tpot_p99": pct(tpot, 99), "n_req": len(ttft)}

    def summarize(self, quiet=False):
        rows = [r for r in self.rows() if r["tag"] == "main"]
        good = [r for r in rows if r.get("failed") == 0 and r.get("output_tok_s") and not r["notes"]]
        arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in good)]
        pr_arm = "mstar-pr-rf" if self.tuned().get("mstar_frontend") == "rust" and "mstar-pr-rf" in arms else "mstar-pr"
        cfg = {a: sorted({r["config"] for r in good if r["arm"] == a}) for a in arms}
        L = [f"### {self.key} ({self.m['hf']}, GPUs {self.m['gpus']})", "",
             "Output tok/s: mean (min-max) over rounds, each round a fresh server process; n = counted rounds. "
             "TTFT/TPOT p50 and p99 in ms, pooled over every request of the counted rounds. "
             f"Configs: {', '.join(f'{a}={c}' for a, c in cfg.items())}", ""]
        present = [w for w in list(WORKLOADS) + sorted({r["workload"] for r in good} - set(WORKLOADS)) if w != "omni_t2s"]
        for wl in present:
            wr = [r for r in good if r["workload"] == wl]
            if not wr:
                continue
            L += [f"#### {wl}", ""]
            head = ["c"] + [f"{a} tok/s" for a in arms] + [f"{pr_arm} vs {a}" for a in arms if a != pr_arm] + \
                   [f"TTFT p50/p99 {a}" for a in arms] + [f"TPOT p50/p99 {a}" for a in arms]
            L += ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
            for c in sorted({r["conc"] for r in wr}):
                means, cells, lat_t, lat_p = {}, [], [], []
                for a in arms:
                    v = [r["output_tok_s"] for r in wr if r["arm"] == a and r["conc"] == c]
                    rr = sorted({r["round"] for r in wr if r["arm"] == a and r["conc"] == c})
                    if v:
                        means[a] = statistics.mean(v)
                        cells.append(f"{means[a]:,.0f} ({min(v):,.0f}-{max(v):,.0f}) n={len(v)}")
                        pl = self.pooled(wl, a, c, rr)
                        lat_t.append(f"{pl['ttft_p50'] * 1e3:.1f}/{pl['ttft_p99'] * 1e3:.1f}" if pl["ttft_p50"] else "-")
                        lat_p.append(f"{pl['tpot_p50'] * 1e3:.2f}/{pl['tpot_p99'] * 1e3:.2f}" if pl["tpot_p50"] else "-")
                    else:
                        cells.append("-")
                        lat_t.append("-")
                        lat_p.append("-")
                vs = []
                for a in arms:
                    if a == pr_arm:
                        continue
                    if pr_arm in means and a in means:
                        vs.append(f"**{(means[pr_arm] / means[a] - 1) * 100:+.1f}%**")
                    else:
                        vs.append("-")
                L.append("| " + " | ".join([str(c)] + cells + vs + lat_t + lat_p) + " |")
            L.append("")
        sp = [r for r in good if r["workload"] == "omni_t2s"] + \
             [r for r in rows if r["workload"] == "omni_t2s" and r.get("failed") == 0 and r.get("audio_s_s")
              and r not in good]
        if sp:
            L += ["#### omni_t2s (text -> speech, benchmark.runner)", "",
                  "audio-s/s and text tok/s: mean (min-max) over rounds; TTFA (first audio) and TTFT (first text) "
                  "p50/p99 and RTF p50: median over rounds of each round's value.", "",
                  "| c | arm | audio s/s | text tok/s | TTFA p50/p99 ms | TTFT p50/p99 ms | RTF p50 | mean audio s |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"]
            for c in sorted({r["conc"] for r in sp}):
                for a in arms + [x for x in ("vllm-omni", "sglang-omni") if x not in arms]:
                    v = [r for r in sp if r["arm"] == a and r["conc"] == c]
                    if not v:
                        continue
                    au = [r["audio_s_s"] for r in v if r.get("audio_s_s")]
                    tx = [r["output_tok_s"] for r in v if r.get("output_tok_s")]

                    def med(k):
                        x = [r[k] for r in v if r.get(k) is not None]
                        return statistics.median(x) if x else None

                    def ms(x):
                        return f"{x * 1e3:.0f}" if x is not None else "-"
                    L.append(f"| {c} | {a} | {statistics.mean(au):.1f} ({min(au):.1f}-{max(au):.1f}) n={len(au)} | "
                             + (f"{statistics.mean(tx):,.0f}" if tx else "-") + f" | {ms(med('ttfa_p50'))}/{ms(med('ttfa_p99'))} | "
                             f"{ms(med('ttft_p50'))}/{ms(med('ttft_p99'))} | {med('rtf_p50') or '-'} | {med('audio_mean_s') or '-'} |"
                             if au else f"| {c} | {a} | - | - | - | - | - | - |")
            L.append("")
        mtp = [r for r in self.rows() if r["tag"] == "mtp" and r.get("failed") == 0 and r.get("output_tok_s")]
        if mtp:
            L += ["#### Supplementary: competitors with their recipe's MTP speculative decoding", "",
                  "Not like-for-like (mstar and the main tables run no speculative decoding); mstar-pr's no-MTP "
                  "numbers are repeated for reference. vLLM: `--speculative-config {\"method\": \"mtp\", "
                  "\"num_speculative_tokens\": 1}`; SGLang: `--speculative-algorithm NEXTN --speculative-num-steps 3 "
                  "--speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.", ""]
            for wl in WORKLOADS:
                wr = [r for r in mtp if r["workload"] == wl]
                if not wr:
                    continue
                L += [f"{wl}:", "", "| c | arm | tok/s mean (min-max) | TTFT p50/p99 ms | TPOT p50/p99 ms |",
                      "|---:|---|---:|---:|---:|"]
                for c in sorted({r["conc"] for r in wr}):
                    for a, tag, src in (("vllm+MTP", "mtp", "vllm"), ("sglang+MTP", "mtp", "sglang"),
                                        (f"{pr_arm} (no MTP)", "main", pr_arm)):
                        pool = wr if tag == "mtp" else [r for r in good if r["workload"] == wl]
                        v = [r["output_tok_s"] for r in pool if r["arm"] == src and r["conc"] == c]
                        if not v:
                            continue
                        rr = sorted({r["round"] for r in pool if r["arm"] == src and r["conc"] == c})
                        pl = self.pooled(wl, src, c, rr, tag)
                        L.append(f"| {c} | {a} | {statistics.mean(v):,.0f} ({min(v):,.0f}-{max(v):,.0f}) n={len(v)} | "
                                 f"{pl['ttft_p50'] * 1e3:.1f}/{pl['ttft_p99'] * 1e3:.1f} | "
                                 f"{pl['tpot_p50'] * 1e3:.2f}/{pl['tpot_p99'] * 1e3:.2f} |")
                L.append("")
        ok_keys = {(r["arm"], r["workload"], r["conc"], r["round"]) for r in good}
        bad = [r for r in rows if r not in good and (r["arm"], r["workload"], r["conc"], r["round"]) not in ok_keys]
        if bad:
            L += ["Excluded cells (failed != 0 or no result):", ""]
            L += [f"- {r['arm']} {r['workload']} c={r['conc']} r{r['round']}: failed={r.get('failed')} "
                  f"{r.get('errors')} {' '.join(r['notes'])}" for r in bad]
            L.append("")
        text = "\n".join(L)
        (self.out / "summary.md").write_text(text + "\n")
        if not quiet:
            print(text)
        return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["data", "correct", "oracle", "tune", "main", "xcheck", "profile", "summarize",
                                      "picktune", "mtp", "fetune", "setfrontend"])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--arms", default="mstar-pr,mstar-main,vllm,sglang")
    ap.add_argument("--rounds", default="1,2,3")
    ap.add_argument("--concs", default=",".join(map(str, CONCS)))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--knobs", default="", help="tune: only these knob names (comma list)")
    ap.add_argument("--arm", default="mstar-pr", help="profile: which M* arm")
    ap.add_argument("--points", default="random_1024_256:128,sharegpt:1",
                    help="profile points wl:c,...")
    a = ap.parse_args()
    b = Bench(a.model)
    arms = a.arms.split(",")
    if a.phase == "data":
        b.phase_data()
    elif a.phase == "correct":
        b.phase_correct(arms)
    elif a.phase == "oracle":
        b.phase_oracle()
    elif a.phase == "tune":
        b.phase_tune([x for x in arms if x in TUNING], [k for k in a.knobs.split(",") if k])
    elif a.phase == "picktune":
        b.pick_tuned()
    elif a.phase == "main":
        b.phase_main([int(x) for x in a.rounds.split(",")], arms, not a.no_rotate,
                     tuple(int(x) for x in a.concs.split(",")), tuple(a.workloads.split(",")))
    elif a.phase == "fetune":
        b.phase_fetune(arms[0])
    elif a.phase == "setfrontend":
        t = b.tuned()
        t["mstar_frontend"] = a.knobs
        (b.out / "tuned.json").write_text(json.dumps(t, indent=1))
    elif a.phase == "mtp":
        b.phase_mtp([x for x in arms if x in Bench.MTP], [int(x) for x in a.rounds.split(",")],
                    tuple(int(x) for x in a.concs.split(",")), tuple(a.workloads.split(",")))
    elif a.phase == "xcheck":
        b.phase_xcheck(arms)
    elif a.phase == "profile":
        b.phase_profile([(p.split(":")[0], int(p.split(":")[1])) for p in a.points.split(",")], a.arm)
    elif a.phase == "summarize":
        b.summarize()


if __name__ == "__main__":
    main()
