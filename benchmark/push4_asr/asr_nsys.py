"""nsys kernel table of a steady-state window of an mstar ASR server (push4a, untimed): the server runs under
`nsys launch`, benchmark.asr_bench drives closed-loop load at one concurrency, and `nsys start/stop` brackets a
window inside it. Writes report.sqlite and kernels.txt (per-process kernel time, omni harness nsys_kernels.py).

  python asr_nsys.py --model nemotron_asr --conc 8 --out ~/push4/results/asr/nsys/nemotron_c8 [--gpu 3]
"""
import argparse
import os
import subprocess
import threading
import time

import asr_gap as g

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--conc", type=int, default=8)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default="3")
ap.add_argument("--cores", default="56-63")
ap.add_argument("--tree", default=f"{g.HOME}/push4/omni/nb")
ap.add_argument("--seconds", type=float, default=4)
a = ap.parse_args()
g.GPU, g.PORT = a.gpu, 28300
os.makedirs(a.out, exist_ok=True)
env = dict(os.environ, HF_HOME=f"{g.HOME}/hf", CUDA_VISIBLE_DEVICES=a.gpu, PYTHONPATH=a.tree, MSTAR_SHM_ARENA="1",
           MSTAR_REQUIRE_ACCELERATOR_GRAPHS="1", TORCHINDUCTOR_CACHE_DIR=f"{g.HOME}/push4/cache/{a.model}/inductor",
           TRITON_CACHE_DIR=f"{g.HOME}/push4/cache/{a.model}/triton", TORCHINDUCTOR_FX_GRAPH_CACHE="1",
           TORCHINDUCTOR_AUTOGRAD_CACHE="1", PATH=f"{g.HOME}/push4/venv/bin:" + os.environ["PATH"])
sess = f"asr{os.getpid()}"
cmd = ["numactl", f"--physcpubind={a.cores}", "nsys", "launch", f"--session-new={sess}", "--trace=cuda,nvtx",
       "--cuda-graph-trace=node", "--trace-fork-before-exec=true", g.VENV_PY, "-m", "mstar.cli.main", "serve", a.model,
       "--config", f"configs/{a.model}.yaml", "--gpus", a.gpu, "--port", str(g.PORT)]
proc = subprocess.Popen(cmd, cwd=a.tree, env=env, stdout=open(f"{a.out}/server.log", "w"), stderr=subprocess.STDOUT,
                        start_new_session=True)


def load(repeats):
    subprocess.run(["numactl", f"--physcpubind={a.cores}", g.VENV_PY, "-m", "benchmark.asr_bench", "--url",
                    f"http://127.0.0.1:{g.PORT}", "--model", a.model, "--system", "mstar", "--data-dir", g.DATA,
                    "--concurrency", str(a.conc), "--repeats", str(repeats), "--warmup", "16"],
                   cwd=a.tree, env=dict(os.environ, HF_HOME=f"{g.HOME}/hf"), capture_output=True, check=False)


try:
    t0 = time.time()
    while not g.healthy():
        if proc.poll() is not None or time.time() - t0 > 3600:
            raise RuntimeError("server did not start")
        time.sleep(5)
    load(1)
    # the fast models finish a pass over the set in about a second: keep the load running through the window
    th = threading.Thread(target=load, args=(40,))
    th.start()
    time.sleep(3)
    subprocess.run(["nsys", "start", f"--session={sess}", "-o", f"{a.out}/report", "--force-overwrite=true"],
                   check=True)
    time.sleep(a.seconds)
    subprocess.run(["nsys", "stop", f"--session={sess}"], check=True)
    th.join()
finally:
    g.stop(proc)
    subprocess.run(["nsys", "shutdown", f"--session={sess}"], capture_output=True, check=False)
subprocess.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", f"{a.out}/report.sqlite",
                f"{a.out}/report.nsys-rep"], check=True)
with open(f"{a.out}/kernels.txt", "w") as f:
    subprocess.run([g.VENV_PY, f"{g.HOME}/push4/omni/harness/nsys_kernels.py", f"{a.out}/report.sqlite", "40"],
                   stdout=f, check=False)
print("NSYS DONE", flush=True)
