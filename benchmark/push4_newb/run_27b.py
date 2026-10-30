"""Qwen3.5-27B TP4 same-box baseline on push4c (GPUs 0-3): profile passes, then the bench with a same-code
noise arm. Resumable: each finished phase leaves <out>/stages/<phase>.done.

  run_27b.py --out ~/push4/results/newb/q27b_baseline [--phases profile,bench]
  run_27b.py --out ~/push4/results/newb/q27b_mix --profile-arms ladder,mix --arms ladder,mix \
      --cells random_1024_256:64,random_1024_256:128 --after-pid <soak pid>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path.home() / "push4" / "newb" / "mstar"
PY = str(Path.home() / "push4" / "newb" / "venv" / "bin" / "python")
LADDER = "benchmark/push4_newb/cfg/q27b_tp4_async_knobs_ladder.yaml"
MIX = "benchmark/push4_newb/cfg/q27b_ladder_mix.yaml"
ARMS = {
    "base": f"base={TREE}",
    "base2": f"base2={TREE}",
    "ladder": f"ladder={TREE}:OVN_CONFIG={LADDER}",
    "mix": f"mix={TREE}:OVN_CONFIG={MIX}",
    "vllm": "vllm=vllm",
    "sglang": "sglang=sglang",
}
COMMON = ["--model", "qwen3_5_27b_tp4", "--gpus", "0,1,2,3", "--port", "59100"]
CELLS = "random_1024_256:32,random_1024_256:64,random_1024_256:128"


def run(phase: str, out: Path, args: list[str]) -> None:
    mark = out / "stages" / f"{phase}.done"
    if mark.exists():
        print(f"skip {phase}", flush=True)
        return
    t0 = time.time()
    print(f"== {phase} {time.strftime('%H:%M:%S')}", flush=True)
    r = subprocess.run([PY, str(HERE / "ab.py"), *args, *COMMON], cwd=str(HERE))
    if r.returncode:
        raise SystemExit(f"{phase} failed rc={r.returncode}")
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(f"{time.time() - t0:.0f}s\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", default="profile,bench")
    ap.add_argument("--arms", default="base,vllm,ladder,sglang,base2")
    ap.add_argument("--profile-arms", default="base,ladder")
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--after-pid", type=int, action="append", default=[],
                    help="wait for these processes (jobs ahead on the same GPUs) to exit first")
    a = ap.parse_args()
    for pid in a.after_pid:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(20)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    phases = a.phases.split(",")
    if "profile" in phases:  # also the cold compile: the first mstar start, untimed
        run("profile", out, ["profile", "--arms", ",".join(ARMS[x] for x in a.profile_arms.split(",")),
                             "--cells", "random_1024_256:128", "--out", str(out / "profile")])
    if "bench" in phases:
        arms = ",".join(ARMS[x] for x in a.arms.split(","))
        run("bench", out, ["bench", "--arms", arms, "--cells", a.cells, "--rounds", str(a.rounds),
                           "--out", str(out / "bench")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
