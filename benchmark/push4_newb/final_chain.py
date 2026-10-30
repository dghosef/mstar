"""push4-newb's final bench on push4c GPUs 0-3: base (d48d4da93) vs head (perf/push4-shared-candidate) vs competitors.

Phase A runs three single-model jobs in parallel on disjoint GPUs, CPUs and ports; phase B runs Qwen3.5-27B TP4 on all
four GPUs, then its gsm8k quality gate and a profile pass. Every job has a hard timeout and its own log, and a failed
job doesn't stop the others. Each mstar tree serves its own configs. Packed mstar servers compile wide and are repinned,
with the API pool sized for the packed CPUs (run_gen.packed_config).

  nohup setsid python final_chain.py >> ~/push4/results/newb/final/chain.log 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/root/push4/newb/venv/bin/python"
OUT = Path.home() / "push4" / "results" / "newb" / "final"
BASE, HEAD = "/root/push4/newb/base", "/root/push4/newb/head"


def evens(a: int, b: int) -> str:
    return ",".join(str(c) for c in range(a, b + 1, 2))


def env(**kv: str) -> dict:
    return {**os.environ, "HF_HOME": "/root/hf", "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}", **kv}


# (name, command, timeout s, env)
PHASE_A = [
    ("droid", [PY, "run_gen.py", "bench", "--model", "droid", "--gpu", "0", "--rounds", "2",
               "--out", str(OUT / "droid"), "--arms", f"base:OVN_TREE={BASE},head:OVN_TREE={HEAD},sglang_b1"],
     3 * 3600, env(PUSH4_MSTAR_CORES="0,2", PUSH4_CORES=evens(4, 40), PUSH4_PORT="59200")),
    ("pi05", [PY, "run_pi05.py", "--gpu", "1", "--rounds", "2", "--n", "50", "--out", str(OUT / "pi05"),
              "--arms", f"base={BASE},head={HEAD},openpi_ws"],
     2 * 3600, env(PUSH4_MSTAR_CORES="42,44", PUSH4_CORES=evens(46, 90), PUSH4_PI05_PORT="59300")),
    ("super_i2v", [PY, "run_gen.py", "bench", "--model", "super_i2v", "--gpu", "2,3", "--rounds", "2",
                   "--out", str(OUT / "super_i2v"), "--arms", f"base:OVN_TREE={BASE},head:OVN_TREE={HEAD},sglang_res"],
     3 * 3600, env(PUSH4_MSTAR_CORES="92,94,96,98", PUSH4_CORES=evens(100, 140), PUSH4_PORT="59400")),
]
TP4 = ["--model", "qwen3_5_27b_tp4", "--gpus", "0,1,2,3", "--port", "59100"]
CELLS = ",".join(f"{w}:{c}" for w in ("random_1024_256", "sharegpt") for c in (32, 64, 128))
PHASE_B = [
    ("q27b", [PY, "ab.py", "bench", "--arms", f"base={BASE},head={HEAD},vllm=vllm,sglang=sglang", "--cells", CELLS,
              "--rounds", "2", "--out", str(OUT / "q27b")] + TP4, 6 * 3600, env(PUSH4_CORES=evens(0, 206))),
    ("q27b_gsm8k", [PY, "ab.py", "gsm8k", "--arms", f"base={BASE},head={HEAD},vllm=vllm", "--n-tok", "250",
                    "--out", str(OUT / "q27b_gsm8k")] + TP4, 2 * 3600, env(PUSH4_CORES=evens(0, 206))),
    ("q27b_profile", [PY, "ab.py", "profile", "--arms", f"head={HEAD}", "--cells", "random_1024_256:128",
                      "--out", str(OUT / "q27b_profile")] + TP4, 3600, env(PUSH4_CORES=evens(0, 206))),
]


def log(msg: str) -> None:
    print(f"[final {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def start(name: str, cmd: list[str], e: dict) -> subprocess.Popen:
    log(f"{name}: start")
    f = open(OUT / f"{name}.log", "w")
    return subprocess.Popen(cmd, cwd=str(HERE), env=e, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def finish(name: str, p: subprocess.Popen, deadline: float) -> None:
    try:
        rc = p.wait(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        rc = "TIMEOUT"
    log(f"{name}: end rc={rc}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    running = [(name, start(name, cmd, e), t0 + timeout) for name, cmd, timeout, e in PHASE_A]
    for name, p, deadline in running:
        finish(name, p, deadline)
    sys.path.insert(0, str(HERE))
    import offline_chain  # leftover servers of this slice, then GPUs 0-3 must be free

    offline_chain.cleanup()
    for name, cmd, timeout, e in PHASE_B:
        p = start(name, cmd, e)
        finish(name, p, time.monotonic() + timeout)
        offline_chain.cleanup()
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
