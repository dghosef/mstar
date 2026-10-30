"""The knobs split on head's shipped Qwen3.5 TP configs: none / kv_plan_cache only / gc_gen0_threshold only / both.

Each arm is head (/root/push4/newb/head) with its shipped config; only the `knobs:` block differs ("both" is the
shipped file itself). 9B TP2 runs as two ab.py processes, one per GPU pair, each running all four arms in opposite
rotation, so every arm gets rounds on both pairs. Then 27B TP4 on all four GPUs. Cells: c=1 and c=128 on random
1024/256 and ShareGPT. mstar servers are packed on 2 CPUs per rank (ab.py PUSH4_MSTAR_CORES: compile wide, repin, API
pool sized for the packed CPUs). Hard timeouts; a failed job doesn't stop the rest.

  nohup setsid python kvpc_chain.py >> ~/push4/results/newb/kvpc/chain.log 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = "/root/push4/newb/venv/bin/python"
HEAD = Path("/root/push4/newb/head")
OUT = Path.home() / "push4" / "results" / "newb" / "kvpc"
CELLS = "random_1024_256:1,random_1024_256:128,sharegpt:1,sharegpt:128"
KNOBS = {"none": None, "kvpc": {"kv_plan_cache": 1}, "gc": {"gc_gen0_threshold": 50000}}


def evens(a: int, b: int) -> str:
    return ",".join(str(c) for c in range(a, b + 1, 2))


def arm_configs(model_cfg: str, tag: str) -> dict[str, str]:
    """name -> config path: the shipped config for "both", copies with another knobs block for the rest."""
    shipped = HEAD / "configs" / model_cfg
    data = yaml.safe_load(shipped.read_text())
    assert data.get("knobs") == {"kv_plan_cache": 1, "gc_gen0_threshold": 50000}, data.get("knobs")
    paths = {"both": str(shipped)}
    for name, knobs in KNOBS.items():
        d = dict(data)
        if knobs is None:
            d.pop("knobs")
        else:
            d["knobs"] = knobs
        path = OUT / "cfg" / f"{tag}_{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(d, sort_keys=False))
        paths[name] = str(path)
    return paths


def arms(paths: dict[str, str], order: list[str]) -> str:
    return ",".join(f"{n}={HEAD}:OVN_CONFIG={paths[n]}" for n in order)


def log(msg: str) -> None:
    print(f"[kvpc {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(jobs: list[tuple[str, list[str], dict, int]]) -> None:
    """Run jobs in parallel; each has its own log and hard timeout."""
    procs = []
    for name, cmd, extra, timeout in jobs:
        log(f"{name}: start")
        f = open(OUT / f"{name}.log", "w")
        p = subprocess.Popen(cmd, cwd=str(HERE), env={**os.environ, "HF_HOME": "/root/hf", **extra}, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, time.monotonic() + timeout))
    for name, p, deadline in procs:
        try:
            rc = p.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            rc = "TIMEOUT"
        log(f"{name}: end rc={rc}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    order = ["none", "kvpc", "gc", "both"]
    p9 = arm_configs("qwen3_5_9b_tp2.yaml", "9b")
    tp2 = ["ab.py", "bench", "--model", "qwen3_5_9b_tp2", "--cells", CELLS, "--rounds", "2"]
    run([
        ("q9b_pairA", [PY, *tp2, "--arms", arms(p9, order), "--gpus", "0,1", "--port", "59100",
                       "--out", str(OUT / "q9b_pairA")],
         {"PUSH4_MSTAR_CORES": evens(0, 6), "PUSH4_CORES": evens(8, 100)}, 4 * 3600),
        ("q9b_pairB", [PY, *tp2, "--arms", arms(p9, order[::-1]), "--gpus", "2,3", "--port", "59150",
                       "--out", str(OUT / "q9b_pairB")],
         {"PUSH4_MSTAR_CORES": evens(102, 108), "PUSH4_CORES": evens(110, 206)}, 4 * 3600),
    ])
    sys.path.insert(0, str(HERE))
    import offline_chain

    offline_chain.cleanup()
    p27 = arm_configs("qwen3_5_27b_tp4.yaml", "27b")
    run([("q27b", [PY, "ab.py", "bench", "--model", "qwen3_5_27b_tp4", "--cells", CELLS, "--rounds", "3",
                   "--arms", arms(p27, order), "--gpus", "0,1,2,3", "--port", "59100", "--out", str(OUT / "q27b")],
          {"PUSH4_MSTAR_CORES": evens(0, 14), "PUSH4_CORES": evens(16, 206)}, 6 * 3600)])
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
