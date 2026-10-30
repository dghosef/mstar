"""Final bench for the omni slice's families on push4a GPUs 2,3 (perf push 4). Arms: base d48d4da93, head 65ee34618
(perf/push4-shared-candidate; Canary's head arm is 65ee34618 + ccb3c770e, its prompt fix) and the best competitor.

  asr    7 ASR models: mstar base / head vs vLLM 0.32 (Whisper-large-v3, Qwen3-ASR 1.7B and 0.6B) or NeMo 3.0.0 offline
         batched transcribe (Parakeet, Canary, Canary-Qwen, Nemotron-ASR); 200 LibriSpeech utterances, c1/8/32/64,
         2 repeats a cell after a 16-request warmup, plus the 605 s long-form file for mstar; 2 rounds, the arm
         order reversed in round 2; then MSTAR_PHASE_TIMING profiles of the head at c8/c32
  rt     Qwen3-ASR realtime (0.6B), mstar only: base vs head, 50 utterances at real time, 1/8/32 sessions, 2 rounds
  omni   Qwen3-Omni audio->text via the omni harness's ab.py: base / head / SGLang-Omni 0.1.9 on short clips
         (c1/4/16/32), ~67 s and ~187 s clips (c1/4/16) and the 4-turn chat (c1/c16), 2 rounds rotated + a profile pass

Resumable: a cell whose result file exists is skipped. Results under ~/push4/results/omni/final/.

  python final_bench.py [--only asr,rt,omni]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import asr_gap as g

HOME = os.path.expanduser("~")
BASE, HEAD, HEAD_CANARY = (f"{HOME}/push4/omni/{t}" for t in ("base", "h65", "h65c"))
FINAL = f"{HOME}/push4/results/omni/final"
g.RES = f"{HOME}/push4/results/omni"  # asr_gap writes RES/<tag>/<engine>_<model>
# (label, mstar model key, competitor engine, competitor key)
ASR = [("Whisper-large-v3", "whisper_large", "vllm", "whisper_large"),
       ("Qwen3-ASR-1.7B", "qwen3_asr", "vllm", "qwen3_asr"),
       ("Qwen3-ASR-0.6B", "qwen3_asr_realtime", "vllm", "qwen3_asr_0.6b"),
       ("Parakeet-TDT-0.6B-v3", "parakeet", "nemo", "parakeet"),
       ("Canary-1B-v2", "canary", "nemo", "canary"),
       ("Canary-Qwen-2.5B", "canary_qwen", "nemo", "canary_qwen"),
       ("Nemotron-ASR", "nemotron_asr", "nemo", "nemotron_asr")]


def log(msg):
    print(f"[final {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def head_tree(model):
    return HEAD_CANARY if model == "canary" else HEAD


def ns(tag, tree, **kw):
    return SimpleNamespace(tag=tag, tree=tree, client_tree=HEAD, config=None, env=kw.get("env", []),
                           conc=kw.get("conc", [1, 8, 32, 64]), repeats=kw.get("repeats", 2),
                           long_form=kw.get("long_form", True), server_cores="32-47", client_cores="48-63")


def done(path):
    try:
        row = json.load(open(path))
        return row.get("rc") == 0 and not row.get("error")
    except (OSError, ValueError):
        return False


def asr_cell(r, arm, model, engine, key):
    if engine == "nemo":
        out = f"{FINAL}/asr/r{r}/nemo/{key}"
        if os.path.exists(f"{out}/nemo.json"):
            return
        log(f"r{r} nemo {key}")
        env = dict(os.environ, HF_HOME=f"{HOME}/hf", CUDA_VISIBLE_DEVICES="2")
        subprocess.run(["numactl", "--physcpubind=32-47", f"{HOME}/push4/engines/nemo/bin/python",
                        f"{HOME}/push4/asr/harness/nemo_bench.py", "--model", key, "--out", out, "--batch", "1", "8",
                        "32", "64", "--repeats", "2"], cwd="/tmp", env=env, check=False,
                       stdout=open(f"{FINAL}/asr/nemo_{key}_r{r}.log", "w"), stderr=subprocess.STDOUT)
        return
    tag = f"final/asr/r{r}/{arm}"
    if done(f"{g.RES}/{tag}/{engine}_{key}/row.json"):
        return
    log(f"r{r} {arm} {engine} {key}")
    tree = head_tree(model) if arm == "head" else BASE
    g.run(engine, key, ns(tag, tree, long_form=engine == "mstar"))


def run_asr():
    for r in (0, 1):
        for _, model, comp, comp_key in ASR:
            arms = [("base", "mstar", model), ("head", "mstar", model), (comp, comp, comp_key)]
            for arm, engine, key in (arms if r == 0 else arms[::-1]):
                asr_cell(r, arm, model, engine, key)
    for _, model, _, _ in ASR:
        tag = "final/asr/prof"
        if not done(f"{g.RES}/{tag}/mstar_{model}/row.json"):
            log(f"profile {model}")
            g.run("mstar", model, ns(tag, head_tree(model), env=["MSTAR_PHASE_TIMING=100"], conc=[8, 32],
                                     repeats=1, long_form=False))


def run_rt():
    for r in (0, 1):
        for arm in (("base", "head") if r == 0 else ("head", "base")):
            out = f"{FINAL}/rt/r{r}/{arm}"
            if os.path.exists(f"{out}/rt.json"):
                continue
            os.makedirs(out, exist_ok=True)
            log(f"rt r{r} {arm}")
            tree = BASE if arm == "base" else HEAD
            env = dict(os.environ, HF_HOME=f"{HOME}/hf", CUDA_VISIBLE_DEVICES=g.GPU, PYTHONUNBUFFERED="1",
                       PYTHONPATH=tree, MSTAR_SHM_ARENA="1", MSTAR_REQUIRE_ACCELERATOR_GRAPHS="1",
                       TORCHINDUCTOR_CACHE_DIR=f"{HOME}/push4/cache/qwen3_asr_realtime/inductor",
                       TRITON_CACHE_DIR=f"{HOME}/push4/cache/qwen3_asr_realtime/triton",
                       TORCHINDUCTOR_FX_GRAPH_CACHE="1", TORCHINDUCTOR_AUTOGRAD_CACHE="1",
                       PATH=f"{HOME}/push4/venv/bin:" + os.environ["PATH"])
            proc = subprocess.Popen(["numactl", "--physcpubind=32-47", g.VENV_PY, "-m", "mstar.cli.main", "serve",
                                     "qwen3_asr_realtime", "--config", "configs/qwen3_asr_realtime.yaml", "--gpus",
                                     g.GPU, "--port", str(g.PORT)], cwd=tree, env=env,
                                    stdout=open(f"{out}/server.log", "w"), stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                t0 = time.time()
                while not g.healthy():
                    if proc.poll() is not None or time.time() - t0 > 2400:
                        raise RuntimeError("server did not start")
                    time.sleep(5)
                res = subprocess.run(["numactl", "--physcpubind=48-63", g.VENV_PY, "-m", "benchmark.asr_realtime_bench",
                                      "--url", f"ws://127.0.0.1:{g.PORT}", "--model", "qwen3_asr_realtime",
                                      "--system", arm, "--data-dir", g.DATA, "--num-utterances", "50", "--sessions",
                                      "1", "8", "32", "--dialect", "openai", "--output-json", f"{out}/rt.json"],
                                     cwd=HEAD, env=dict(os.environ, HF_HOME=f"{HOME}/hf", HF_HUB_OFFLINE="1"),
                                     capture_output=True, text=True, check=False)
                open(f"{out}/client.log", "w").write(res.stdout + res.stderr)
            except RuntimeError as e:
                log(f"rt {arm}: {e}")
            finally:
                g.stop(proc)


def run_omni():
    if os.path.exists(f"{g.RES}/final/omni/done"):
        return
    log("omni ab")
    cells = ("a2t:1,a2t:4,a2t:16,a2t:32,a2tlong60:1:32,a2tlong60:4:32,a2tlong60:16:32,a2tlong180:1:32,"
             "a2tlong180:4:32,a2tlong180:16:32,mta2t:1:8,mta2t:16:32")
    rc = subprocess.run([g.VENV_PY, f"{HOME}/push4/omni/harness/ab.py", "--tag", "final/omni", "--rounds", "2",
                         "--warm", "1", "--cells", cells, "--arm", f"base={BASE}", "--arm", f"head={HEAD}",
                         "--arm", "sgomni=SGOMNI", "--arm-serve", "base=--enable-metrics", "--arm-serve",
                         "head=--enable-metrics", "--profile-cells", "a2t:16,a2tlong180:4:32"],
                        env=dict(os.environ, PATH=f"{HOME}/push4/venv/bin:{HOME}/.local/bin:" + os.environ["PATH"]),
                        check=False).returncode
    if rc == 0:
        open(f"{g.RES}/final/omni/done", "w").write("ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="asr,rt,omni")
    a = ap.parse_args()
    os.makedirs(f"{FINAL}/asr", exist_ok=True)
    steps = {"asr": run_asr, "rt": run_rt, "omni": run_omni}
    for step in a.only.split(","):
        steps[step]()
    log("FINAL BENCH DONE")


if __name__ == "__main__":
    sys.exit(main())
