"""Warm the compile caches for mstar ASR servers off the timed GPU: start each model's server on GPU 3 (4 cores)
until it is healthy, then stop it. Untimed; the timed arms then start warm.

  python precompile.py --models qwen3_asr,canary --cache-root ~/push4/asr  [--tree TREE]
"""
import argparse
import os
import subprocess
import time

import asr_gap as g

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--tree", default=f"{g.HOME}/push4/omni/nb")
ap.add_argument("--cache-root", default=None, help="Inductor/Triton cache root (default: per model, as asr_gap)")
a = ap.parse_args()
g.GPU, g.PORT, g.SERVER_CORES = "3", 28300, "60-63"
for model in a.models.split(","):
    root = a.cache_root or f"{g.HOME}/push4/cache/{model}"
    env = dict(os.environ, HF_HOME=f"{g.HOME}/hf", CUDA_VISIBLE_DEVICES=g.GPU, PYTHONPATH=a.tree, MSTAR_SHM_ARENA="1",
               MSTAR_REQUIRE_ACCELERATOR_GRAPHS="1", TORCHINDUCTOR_CACHE_DIR=f"{root}/inductor",
               TRITON_CACHE_DIR=f"{root}/triton", TORCHINDUCTOR_FX_GRAPH_CACHE="1", TORCHINDUCTOR_AUTOGRAD_CACHE="1",
               TORCHINDUCTOR_COMPILE_THREADS="4", PATH=f"{g.HOME}/push4/venv/bin:" + os.environ["PATH"])
    cmd = [g.VENV_PY, "-m", "mstar.cli.main", "serve", model, "--config", f"configs/{model}.yaml", "--gpus", g.GPU,
           "--port", str(g.PORT)]
    t0 = time.time()
    log = open(f"{g.RES}/precompile_{model}.log", "w")
    proc = subprocess.Popen(["numactl", f"--physcpubind={g.SERVER_CORES}"] + cmd, cwd=a.tree, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    while not g.healthy() and proc.poll() is None and time.time() - t0 < 3600:
        time.sleep(5)
    ok = g.healthy()
    g.stop(proc)
    print(f"[precompile] {model}: {'ready' if ok else 'FAILED'} after {round(time.time() - t0)} s", flush=True)
