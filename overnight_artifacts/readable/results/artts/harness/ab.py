#!/usr/bin/env python3
"""Run bench_arm.py for alternating arms, sequentially.

    python ab.py --arms base=/root/work/base,chg=/root/work/mstar --rounds 2 --start-round 1 \
        --out /workspace/results/q06_ab1 -- --model qwen3_tts --config configs/qwen3tts.yaml ...

Everything after ``--`` is passed to each bench_arm.py call. Each arm can carry
extra bench_arm args after a ``|``: ``bf16=/root/work/base|--extra-arg ...`` is
not supported; use separate invocations for that.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

argv = sys.argv[1:]
rest = []
if "--" in argv:
    i = argv.index("--")
    argv, rest = argv[:i], argv[i + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("--arms", required=True)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--start-round", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--arm-env", action="append", default=[], help="arm:KEY=VAL")
args = ap.parse_args(argv)
arms = [a.split("=", 1) for a in args.arms.split(",")]
for r in range(args.start_round, args.start_round + args.rounds):
    order = arms if r % 2 else list(reversed(arms))
    for name, tree in order:
        extra = []
        for ae in args.arm_env:
            an, _, kv = ae.partition(":")
            if an == name:
                extra += ["--env", kv]
        cmd = [sys.executable, str(HERE / "bench_arm.py"), "--tree", tree, "--label", f"{name}_r{r}",
               "--out", args.out, *rest, *extra]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd)
print("AB DONE", flush=True)
