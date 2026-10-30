"""ASR gap table on push4a (asr slice: GPU 2, server cores 32-47, client cores 48-63): one fresh server per
(engine, model), benchmark.asr_bench closed loop over the 200-utterance LibriSpeech set (RTFx, latency, WER),
results/asr/<tag>/<engine>_<model>/bench.json.

  python asr_gap.py --tag gap1 --runs mstar:whisper_large,vllm:whisper_large --conc 1 8 32 64 [--tree TREE]
      [--env K=V] [--config-override path] [--repeats 2]

mstar serves `configs/<key>.yaml` from --tree; vLLM 0.32 serves the HF checkpoint (bf16, async scheduling, 64 seqs).
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

HOME = os.path.expanduser("~")
VENV_PY = f"{HOME}/push4/venv/bin/python"
VLLM_BIN = f"{HOME}/push4/engines/vllm/bin"
RES = f"{HOME}/push4/results/asr"
DATA = f"{HOME}/push4/asr/data"
GPU = "2"
SERVER_CORES, CLIENT_CORES = "32-47", "48-63"
PORT = 28200
HF = {"whisper_large": "openai/whisper-large-v3", "qwen3_asr": "Qwen/Qwen3-ASR-1.7B",
      "qwen3_asr_realtime": "Qwen/Qwen3-ASR-0.6B", "qwen3_asr_0.6b": "Qwen/Qwen3-ASR-0.6B",
      "canary": "nvidia/canary-1b-v2",
      "canary_qwen": "nvidia/canary-qwen-2.5b", "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
      "nemotron_asr": "nvidia/nemotron-3.5-asr-streaming-0.6b"}
VLLM_ARGS = {"qwen3_asr_realtime": ["--hf-overrides", '{"architectures": ["Qwen3ASRRealtimeGeneration"]}']}


def healthy() -> bool:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=2) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:  # noqa: BLE001
            pass
    return False


def gpu_mem() -> int:
    out = subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout
    return int(out.split()[0])


def stop(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGKILL, 0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        for _ in range(wait):
            if proc.poll() is not None:
                break
            time.sleep(1)
    subprocess.run(["pkill", "-9", "-s", str(proc.pid)], capture_output=True, check=False)
    for _ in range(90):
        if gpu_mem() < 2000:
            break
        time.sleep(2)


def run(engine: str, model: str, a: argparse.Namespace) -> dict:
    out = f"{RES}/{a.tag}/{engine}_{model}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, HF_HOME=f"{HOME}/hf", CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED="1")
    env.update(dict(kv.split("=", 1) for kv in a.env))
    if engine == "mstar":
        env.update({"PYTHONPATH": a.tree, "MSTAR_SHM_ARENA": "1", "MSTAR_REQUIRE_ACCELERATOR_GRAPHS": "1",
                    # persistent per-model compile caches shared by every arm and round (COMPILE_CACHE_NOTE)
                    "TORCHINDUCTOR_CACHE_DIR": f"{HOME}/push4/cache/{model}/inductor",
                    "TRITON_CACHE_DIR": f"{HOME}/push4/cache/{model}/triton",
                    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
                    "PATH": f"{HOME}/push4/venv/bin:{HOME}/.local/bin:" + env["PATH"]})
        config = a.config or f"configs/{model}.yaml"
        cmd = [VENV_PY, "-m", "mstar.cli.main", "serve", model, "--config", config, "--gpus", GPU, "--port", str(PORT)]
        cwd = a.tree
    else:
        env["PATH"] = f"{VLLM_BIN}:" + env["PATH"]
        cmd = [f"{VLLM_BIN}/vllm", "serve", HF[model], "--port", str(PORT), "--dtype", "bfloat16", "--max-num-seqs",
               "64", "--gpu-memory-utilization", "0.6", "--async-scheduling"] + VLLM_ARGS.get(model, [])
        cwd = HOME
    assert not healthy(), f"port {PORT} already serving"
    assert gpu_mem() < 2000, f"GPU {GPU} busy"
    t0 = time.time()
    slog = open(f"{out}/server.log", "w")
    proc = subprocess.Popen(["numactl", f"--physcpubind={a.server_cores}"] + cmd, cwd=cwd, env=env, stdout=slog,
                            stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[gap] {engine} {model} starting: {' '.join(cmd)}", flush=True)
    row = {"engine": engine, "model": model, "tree": a.tree if engine == "mstar" else None, "env": a.env}
    try:
        while not healthy():
            if proc.poll() is not None:
                raise RuntimeError(f"server died rc={proc.returncode}; see {out}/server.log")
            if time.time() - t0 > 2400:
                raise RuntimeError("startup timeout")
            time.sleep(5)
        row["startup_s"] = round(time.time() - t0)
        print(f"[gap] ready after {row['startup_s']} s", flush=True)
        bench = ["numactl", f"--physcpubind={a.client_cores}", VENV_PY, "-m", "benchmark.asr_bench", "--url",
                 f"http://127.0.0.1:{PORT}", "--model", HF[model] if engine == "vllm" else model, "--system",
                 engine, "--data-dir", DATA, "--concurrency", *map(str, a.conc), "--repeats", str(a.repeats),
                 "--warmup", "16", "--output-json", f"{out}/bench.json"]
        r = subprocess.run(bench, cwd=a.client_tree, env=dict(os.environ, HF_HOME=f"{HOME}/hf"),
                           capture_output=True, text=True, timeout=7200, check=False)
        open(f"{out}/client.log", "w").write(r.stdout + r.stderr)
        row["rc"] = r.returncode
        print(r.stdout[-1500:], flush=True)
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)
        print(f"[gap] {engine} {model} FAILED: {e!r}", flush=True)
    finally:
        stop(proc)
    if engine == "mstar" and "MSTAR_PHASE_TIMING" in env:
        render = ("from benchmark.worker_phases.phases import parse_log, render, segment; import sys; "
                  "print(render(segment(parse_log(open(sys.argv[1]).read()))))")
        r = subprocess.run([VENV_PY, "-c", render, f"{out}/server.log"], cwd=a.tree, capture_output=True, text=True,
                           check=False, env=dict(os.environ, PYTHONPATH=a.tree))
        open(f"{out}/phases.txt", "w").write(r.stdout + r.stderr[-2000:])
    json.dump(row, open(f"{out}/row.json", "w"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", required=True, help="engine:model,...")
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tree", default=f"{HOME}/push4/omni/nb")
    ap.add_argument("--client-tree", default=f"{HOME}/push4/omni/nb")
    ap.add_argument("--config", default=None)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--server-cores", default=SERVER_CORES)
    ap.add_argument("--client-cores", default=CLIENT_CORES)
    a = ap.parse_args()
    for spec in a.runs.split(","):
        engine, model = spec.split(":")
        run(engine, model, a)
    print("[gap] ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
