"""Perf push 4, newb A/B driver (runs on push4c, GPUs 0-3, NUMA 0). Copied from push4_q35text/ab.py.

Arms: ``name=/path/to/tree[:K=V;K=V]`` for mstar trees (sharing ~/push4/venv, the tree's own mstar/ wins via
PYTHONPATH), or ``vllm`` / ``sglang`` (``name=vllm[:K=V]``) for the competitors with the Oct-21 tuned flags.
Every process runs under ``numactl --physcpubind=$PUSH4_CORES`` (default 0-31) with CUDA_VISIBLE_DEVICES=--gpus.
mstar servers get MSTAR_SHM_ARENA=1 and MSTAR_REQUIRE_ACCELERATOR_GRAPHS=1, like the Oct-21 benchmark. The
client is the benchmark's own client.py (OpenAI chat, stream, ignore_eos, fixed max_tokens per row).

Special env keys in an mstar arm: OVN_SERVE_EXTRA (extra serve args), OVN_CONFIG (config path relative to the
tree, or absolute), OVN_WRAP (a command prefix, e.g. nsys). In a competitor arm: OVN_SERVE_EXTRA too.

  ab.py bench   --arms base=~/push4/q35text/base,new=~/push4/q35text/mstar --model qwen3_5_0.8b \
                --cells random_1024_256:16,random_1024_256:64 --rounds 2 --out ~/push4/results/q35text/x
  ab.py profile --arms ... --cells random_1024_256:64 --out ...   (MSTAR_PHASE_TIMING=100)
  ab.py correct --arms ... --out ...                               (greedy texts per arm, compared)
  ab.py gsm8k   --arms ... --n-tok 250 --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
# own envs (push4c's shared setup died 2026-10-29): ~/push4/newb/{venv,engines}
VENV = Path(os.environ.get("OVN_VENV", HOME / "push4" / "newb" / "venv"))
PY = str(VENV / "bin" / "python")
ENGINES = Path(os.environ.get("OVN_ENGINES", HOME / "push4" / "newb" / "engines"))
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("OVN_DATA", HOME / "push4" / "newb" / "data"))
HF = str(HOME / "hf")
# push4c: NUMA 0 = even CPUs + GPUs 0-3 (NUMA ids interleave by parity)
CORES = os.environ.get("PUSH4_CORES", ",".join(str(c) for c in range(0, 207, 2)))
MEMNODE = os.environ.get("PUSH4_MEMNODE", "")
CACHE = HOME / "push4" / "newb"
CONFIGS = {
    "qwen3_5_0.8b": "configs/qwen3_5_0.8b.yaml",
    "qwen3_5_2b": "configs/qwen3_5_2b.yaml",
    "qwen3_5_4b": "configs/qwen3_5_4b.yaml",
    "qwen3_5_9b": "configs/qwen3_5_9b.yaml",
    "qwen3_5_9b_tp2": "configs/qwen3_5_9b_tp2.yaml",
    "qwen3_5_27b_tp4": "configs/qwen3_5_27b_tp4.yaml",
}
# Oct-21 tuned competitor flags (bench1021 results/q35_*/tuned.json); 4B was never tuned: it takes 2B's.
TUNED = {
    "vllm": {"qwen3_5_0.8b": ["--language-model-only"], "qwen3_5_2b": ["--language-model-only"],
             "qwen3_5_4b": ["--language-model-only"], "qwen3_5_9b": ["--language-model-only"],
             "qwen3_5_9b_tp2": [], "qwen3_5_27b_tp4": ["--language-model-only"]},
    "sglang": {"qwen3_5_0.8b": ["--attention-backend", "fa3"],
               "qwen3_5_2b": ["--chunked-prefill-size", "16384"],
               "qwen3_5_4b": ["--chunked-prefill-size", "16384"],
               "qwen3_5_9b": ["--chunked-prefill-size", "16384"],
               "qwen3_5_9b_tp2": ["--chunked-prefill-size", "16384"],
               "qwen3_5_27b_tp4": ["--chunked-prefill-size", "16384"]},
}


def log(msg: str) -> None:
    print(f"[ab {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_arms(spec: str) -> list[tuple[str, str, dict]]:
    arms = []
    for part in spec.split(","):
        name, rest = part.split("=", 1)
        path, _, envs = rest.partition(":")
        env = dict(kv.replace("|", ",").split("=", 1) for kv in envs.split(";") if kv) if envs else {}
        arms.append((name, os.path.expanduser(path), env))
    return arms


def tp_of(model: str) -> int:
    for part in model.split("_"):
        if part.startswith("tp"):
            return int(part[2:])
    return 1


def model_id(model: str) -> str:
    return f"Qwen/Qwen3.5-{model.split('_')[2].upper()}"


def pin(cmd: list[str]) -> list[str]:
    # the Runpod container may not set_mempolicy (no CAP_SYS_NICE): membind only when PUSH4_MEMNODE is set;
    # first-touch keeps memory local to the pinned CPUs anyway
    mem = [f"--membind={MEMNODE}"] if MEMNODE else []
    return ["numactl", f"--physcpubind={CORES}", *mem] + cmd


def base_env(gpus: str) -> dict:
    env = dict(os.environ)
    env.update({"HF_HOME": HF, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": gpus,
                "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "inductor"), "TRITON_CACHE_DIR": str(CACHE / "triton"),
                "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
                # push4a's fabric manager can't bind NVLS multicast memory: NCCL init fails with CUDA error 401
                "NCCL_NVLS_ENABLE": "0"})
    return env


def mstar_env(tree: str, extra: dict, gpus: str) -> dict:
    env = base_env(gpus)
    env.update({
        "PYTHONPATH": tree,
        "PATH": f"{VENV / 'bin'}:{env.get('PATH', '')}",
        "VIRTUAL_ENV": str(VENV),
        "MSTAR_SHM_ARENA": "1",
        "MSTAR_REQUIRE_ACCELERATOR_GRAPHS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "8",
        "MAX_JOBS": "8",
    })
    env.update({k: v for k, v in extra.items() if not k.startswith("OVN_") or k == "OVN_STEPLOG"})
    if extra.get("OVN_HOOKS") or extra.get("OVN_STEPLOG") or extra.get("MSTAR_TPROF"):
        env["PYTHONPATH"] = f"{HERE / 'hooks'}:{tree}"
    return env


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def competitor_cmd(kind: str, model: str, port: int, extra: dict) -> list[str]:
    tp = tp_of(model)
    mid = model_id(model)
    if kind == "vllm":
        cmd = [str(ENGINES / "vllm" / "bin" / "vllm"), "serve", mid, "--tensor-parallel-size", str(tp),
               "--max-model-len", "32768", "--max-num-seqs", "128", "--no-enable-prefix-caching",
               "--gpu-memory-utilization", "0.9", "--host", "127.0.0.1", "--port", str(port)]
    else:
        cmd = [str(ENGINES / "sglang" / "bin" / "python"), "-m", "sglang.launch_server", "--model-path", mid,
               "--tp", str(tp), "--context-length", "32768", "--max-running-requests", "128",
               "--disable-radix-cache", "--mamba-ssm-dtype", "float32", "--mem-fraction-static", "0.85",
               "--host", "127.0.0.1", "--port", str(port)]
    cmd += TUNED[kind][model]
    if extra.get("OVN_SERVE_EXTRA"):
        cmd += extra["OVN_SERVE_EXTRA"].split()
    return cmd


class Server:
    def __init__(self, name, tree, env_extra, model, gpus, port, logpath):
        self.port = port
        if healthy(port):
            raise RuntimeError(f"port {port} already serving")
        self.kind = tree if tree in ("vllm", "sglang") else "mstar"
        if self.kind == "mstar":
            env = mstar_env(tree, env_extra, gpus)
            serve_model = "_".join(model.split("_")[:3])  # qwen3_5_9b_tp2 -> qwen3_5_9b
            cfg = env_extra.get("OVN_CONFIG", CONFIGS[model])
            config = cfg if cfg.startswith("/") else str(Path(tree) / cfg)
            cmd = [PY, "-m", "mstar.cli.main", "serve", serve_model, "--config", config, "--gpus", gpus,
                   "--port", str(port), "--socket-path-prefix", f"/tmp/p4n_{port}_{int(time.time())}/"]
            if env_extra.get("OVN_SERVE_EXTRA"):
                cmd += env_extra["OVN_SERVE_EXTRA"].split()
            cwd = tree
        else:
            env = base_env(gpus)
            # the engine's own venv first: SGLang's JIT kernels shell out to its `ninja`
            env["PATH"] = f"{ENGINES / self.kind / 'bin'}:{env.get('PATH', '')}"
            # push4c's host toolkit is CUDA 12.8; SGLang's JIT kernels (deep_gemm, custom all-reduce) need
            # nvcc >= 12.9: use a CUDA 13.0 toolkit like push4a's host one (runfile, --toolkitpath)
            cu13 = HOME / "push4" / "newb" / "cuda-13.0"
            if (cu13 / "bin" / "nvcc").exists():
                env["CUDA_HOME"] = str(cu13)
                env["PATH"] = f"{cu13 / 'bin'}:{env['PATH']}"
            env["SGLANG_JIT_CACHE_DIR"] = str(CACHE / "sglang_jit")  # own JIT cache (others share ~/.cache)
            env.update({k: v for k, v in env_extra.items() if not k.startswith("OVN_")})
            cmd = competitor_cmd(self.kind, model, port, env_extra)
            cwd = str(CACHE)
        cmd = pin(cmd)
        if env_extra.get("OVN_WRAP"):  # e.g. an nsys launch prefix
            cmd = env_extra["OVN_WRAP"].split() + cmd
        self.logf = open(logpath, "w")
        self.logf.write(f"# arm={name} tree={tree} env={env_extra} cmd={' '.join(cmd)}\n")
        self.logf.flush()
        self.p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=self.logf, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        t0 = time.monotonic()
        self.wrapped = bool(env_extra.get("OVN_WRAP"))
        while time.monotonic() - t0 < 3600:
            if self.p.poll() is not None:
                raise RuntimeError(f"server {name} died at startup; see {logpath}")
            if healthy(port):
                self.startup_s = time.monotonic() - t0
                log(f"server {name} up in {self.startup_s:.0f}s")
                return
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"server {name} not healthy in 60 min")

    def stop(self) -> None:
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGINT)
                self.p.wait(timeout=600 if self.wrapped else 90)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait(timeout=30)
            except ProcessLookupError:
                pass
        subprocess.run(["pkill", "-KILL", "-s", str(self.p.pid)], capture_output=True)
        t0 = time.monotonic()
        while healthy(self.port) and time.monotonic() - t0 < 60:
            time.sleep(1)
        time.sleep(5)
        self.logf.close()


def gpu_mem(gpus: str) -> list[int]:
    r = subprocess.run(["nvidia-smi", f"--id={gpus}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


def run_cell(model: str, port: int, wl: str, conc: int, n: int, warm: int, outdir: Path, gpus: str = "") -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    mid = model_id(model)
    cmd = pin([PY, "-I", str(HERE / "client.py"), "--url", f"http://127.0.0.1:{port}", "--model-id", mid,
               "--data", str(DATA / f"{wl}.jsonl"), "--conc", str(conc), "--num", str(n),
               "--warmup-waves", str(warm), "--warmup-offset", "1100", "--tokenizer", mid, "--out", str(outdir)])
    load0 = os.getloadavg()[0]
    r = subprocess.run(cmd, env={**os.environ, "HF_HOME": HF, "HF_HUB_OFFLINE": "1"},
                       capture_output=True, text=True, timeout=3600)
    (outdir / "client.log").write_text(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
    row = {"wl": wl, "conc": conc, "n": n, "rc": r.returncode, "load0": round(load0, 1),
           "load1": round(os.getloadavg()[0], 1)}
    if gpus:
        row["gpu_mem_mib"] = gpu_mem(gpus)
    rj = outdir / "results.json"
    if rj.exists():
        s = json.loads(rj.read_text())["summary"]
        for k in ("completed", "failed", "output_tok_s", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
                  "retok_ratio", "errors"):
            row[k] = s.get(k)
    return row


def parse_cells(spec: str) -> list[tuple[str, int]]:
    return [(c.split(":")[0], int(c.split(":")[1])) for c in spec.split(",")]


def cmd_bench(a) -> int:
    arms = parse_arms(a.arms)
    out = Path(a.out)
    (out / "servers").mkdir(parents=True, exist_ok=True)
    rows = out / "bench.jsonl"
    cells = parse_cells(a.cells)
    for r in range(a.first_round, a.first_round + a.rounds):
        order = arms if r % 2 else arms[::-1]
        for name, tree, env in order:
            s = Server(name, tree, env, a.model, a.gpus, a.port, out / "servers" / f"{name}_r{r}.log")
            try:
                for i, (wl, c) in enumerate(cells):
                    n = max(a.min_n, a.mult * c)
                    warm = a.warm_first if i == 0 else 1
                    row = run_cell(a.model, a.port, wl, c, n, warm, out / "cells" / f"{name}_r{r}_{wl}_c{c}",
                                   a.gpus)
                    row.update({"arm": name, "round": r, "model": a.model, "ts": time.time(),
                                "startup_s": round(s.startup_s)})
                    with open(rows, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    log(f"r{r} {name} {wl} c={c}: {row.get('output_tok_s') or 0:.0f} tok/s "
                        f"ttft p50/p99 {row.get('ttft_p50')}/{row.get('ttft_p99')} failed={row.get('failed')} "
                        f"load={row['load0']}->{row['load1']}")
            finally:
                s.stop()
    summarize(out)
    return 0


def render_phases(path: Path, tree: str) -> str:
    sys.path.insert(0, tree)
    from benchmark.worker_phases.phases import parse_log, render, segment

    return render(segment(parse_log(path.read_text())))


def cmd_profile(a) -> int:
    arms = parse_arms(a.arms)
    out = Path(a.out)
    (out / "servers").mkdir(parents=True, exist_ok=True)
    (out / "profile").mkdir(parents=True, exist_ok=True)
    for name, tree, env in arms:
        for wl, c in parse_cells(a.cells):
            lp = out / "servers" / f"{name}_profile_{wl}_c{c}.log"
            s = Server(name, tree, {**env, "MSTAR_PHASE_TIMING": "100"}, a.model, a.gpus, a.port, lp)
            try:
                row = run_cell(a.model, a.port, wl, c, max(a.min_n, a.mult * c), 2,
                               out / "cells" / f"{name}_profile_{wl}_c{c}")
                log(f"profile {name} {wl} c={c}: {row.get('output_tok_s') or 0:.0f} tok/s (timing on)")
            finally:
                s.stop()
            try:
                (out / "profile" / f"{name}_{wl}_c{c}.phases.txt").write_text(render_phases(lp, tree))
            except Exception as e:  # noqa: BLE001
                log(f"render failed: {e}")
    return 0


CORRECT_PROMPTS = [
    "What is the capital of France? Talk about the city.",
    "Explain the difference between machine learning and deep learning.",
    "Write 5 haikus about autumn leaves.",
    "List the first 20 prime numbers and explain how you found them.",
    "Translate 'The quick brown fox jumps over the lazy dog' into French, German and Spanish.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What are the main causes of climate change?",
]


def _greedy(port: int, model: str, n_tok: int, conc: int, prompts: list[str]) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        body = {"model": model_id(model), "messages": [{"role": "user", "content": p}], "temperature": 0,
                "max_tokens": n_tok, "stream": False, "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    with ThreadPoolExecutor(conc) as ex:
        return list(ex.map(one, prompts))


def cmd_correct(a) -> int:
    arms = parse_arms(a.arms)
    out = Path(a.out)
    (out / "servers").mkdir(parents=True, exist_ok=True)
    prompts = list(CORRECT_PROMPTS)
    # long prompts too: the random dataset's first rows (prefix + question), so long-prefill paths run
    rows = [json.loads(x) for x in (DATA / "random_1024_256.jsonl").read_text().splitlines()[:8]]
    longp = [r["prompt"] + "\n\nIgnore the text above. " + q for r, q in zip(rows, CORRECT_PROMPTS, strict=True)]
    texts = {}
    for name, tree, env in arms:
        s = Server(name, tree, env, a.model, a.gpus, a.port, out / "servers" / f"{name}_correct.log")
        try:
            texts[f"{name}_c1"] = _greedy(a.port, a.model, a.n_tok, 1, prompts)
            texts[f"{name}_c1b"] = _greedy(a.port, a.model, a.n_tok, 1, prompts)
            texts[f"{name}_c8"] = _greedy(a.port, a.model, a.n_tok, 8, prompts)
            texts[f"{name}_long_c1"] = _greedy(a.port, a.model, a.n_tok, 1, longp)
            texts[f"{name}_long_c8"] = _greedy(a.port, a.model, a.n_tok, 8, longp)
        finally:
            s.stop()
    (out / "correct.json").write_text(json.dumps(texts, indent=1))
    keys = list(texts)
    lines = []
    for i, k in enumerate(keys):
        for k2 in keys[i + 1:]:
            if ("long" in k) != ("long" in k2):
                continue
            same = sum(x == y for x, y in zip(texts[k], texts[k2], strict=True))
            pre = [len(os.path.commonprefix([x, y])) for x, y in zip(texts[k], texts[k2], strict=True)]
            lines.append(f"{k} vs {k2}: identical {same}/{len(texts[k])}  common-prefix chars {pre}")
    (out / "correct.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def cmd_gsm8k(a) -> int:
    arms = parse_arms(a.arms)
    out = Path(a.out)
    (out / "servers").mkdir(parents=True, exist_ok=True)
    for name, tree, env in arms:
        s = Server(name, tree, env, a.model, a.gpus, a.port, out / "servers" / f"{name}_gsm8k.log")
        try:
            r = subprocess.run(pin([PY, str(HERE / "gsm8k.py"), "--url", f"http://127.0.0.1:{a.port}",
                                    "--model-id", model_id(a.model), "--n", str(a.n_tok), "--conc", "32",
                                    "--out", str(out / f"gsm8k_{name}.json")]),
                               env={**os.environ, "HF_HOME": HF}, capture_output=True, text=True)
            (out / f"gsm8k_{name}.log").write_text(r.stdout[-5000:] + r.stderr[-5000:])
            log(f"gsm8k {name}: {r.stdout.strip().splitlines()[-1:] if r.stdout.strip() else r.stderr[-300:]}")
        finally:
            s.stop()
    return 0


def summarize(out: Path) -> None:
    rows = [json.loads(x) for x in (out / "bench.jsonl").read_text().splitlines() if x.strip()]
    by: dict = {}
    for r in rows:
        if r.get("output_tok_s") and not r.get("failed"):
            by.setdefault((r["model"], r["wl"], r["conc"], r["arm"]), []).append(r)
    lines = ["| model | wl | c | arm | tok/s mean | sd | runs | ttft p50 ms | ttft p99 ms | tpot p50 ms | load |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (m, wl, c, arm), v in sorted(by.items()):
        t = [x["output_tok_s"] for x in v]
        sd = statistics.stdev(t) if len(t) > 1 else 0.0
        ttft = statistics.mean(x["ttft_p50"] for x in v) * 1000
        ttft99 = statistics.mean(x["ttft_p99"] for x in v) * 1000
        tpot = statistics.mean(x["tpot_p50"] for x in v) * 1000
        load = [x.get("load0") for x in v]
        lines.append(f"| {m} | {wl} | {c} | {arm} | {statistics.mean(t):.0f} | {sd:.0f} | "
                     f"{[round(x) for x in t]} | {ttft:.1f} | {ttft99:.1f} | {tpot:.2f} | {load} |")
    bad = [r for r in rows if r.get("failed") or not r.get("output_tok_s")]
    if bad:
        lines.append(f"\nBAD CELLS: {[(r['arm'], r['round'], r['wl'], r['conc'], r.get('failed')) for r in bad]}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["bench", "profile", "correct", "summarize", "gsm8k"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--model", default="qwen3_5_0.8b")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--port", type=int, default=59100)
    ap.add_argument("--cells", default="random_1024_256:16")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--first-round", type=int, default=1)
    ap.add_argument("--mult", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=16)
    ap.add_argument("--warm-first", type=int, default=3)
    ap.add_argument("--n-tok", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.out = os.path.expanduser(a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.phase == "summarize":
        summarize(Path(a.out))
        return 0
    return {"bench": cmd_bench, "profile": cmd_profile, "correct": cmd_correct, "gsm8k": cmd_gsm8k}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
