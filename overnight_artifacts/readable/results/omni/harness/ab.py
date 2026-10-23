#!/usr/bin/env python3
"""Alternate arms (A B A B ...) through bench.py, then WER on every t2s/a2s dir.

  python ab.py --tag ab1 --rounds 2 --workloads t2s_c1,t2s_c8 \
      --arm base=/root/work/base --arm chg=/root/work/mstar [--arm-env chg=K=V] [--no-wer]
"""
import argparse
import glob
import os
import subprocess
import sys

H = os.environ.get("OMNI_HARNESS", "/root/work/harness_omni")
PY = "/root/work/mstar/.venv/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workloads", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=tree")
    ap.add_argument("--arm-env", action="append", default=[], help="name=K=V")
    ap.add_argument("--config", default="configs/qwen3omni_2gpu.yaml")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--port", default="8100")
    ap.add_argument("--phase-timing", type=int, default=100)
    ap.add_argument("--no-wer", action="store_true")
    args = ap.parse_args()
    arms = [a.split("=", 1) for a in args.arm]
    envs = {}
    for e in args.arm_env:
        name, kv = e.split("=", 1)
        envs.setdefault(name, []).append(kv)
    for r in range(args.rounds):
        order = arms if r % 2 == 0 else arms[::-1]
        for name, tree in order:
            cmd = [PY, f"{H}/bench.py", "--tree", tree, "--arm", f"{name}_r{r}", "--tag", args.tag,
                   "--workloads", args.workloads, "--phase-timing", str(args.phase_timing),
                   "--config", args.config, "--gpus", args.gpus, "--port", args.port]
            for kv in envs.get(name, []):
                cmd += ["--env", kv]
            print("+", " ".join(cmd), flush=True)
            rc = subprocess.run(cmd).returncode
            print(f"[ab] {name} r{r} rc={rc}", flush=True)
    if not args.no_wer:
        dirs = sorted(d for d in glob.glob(f"/workspace/results/{args.tag}/*/*")
                      if ("t2s" in d or "a2s" in d) and glob.glob(f"{d}/*.wav"))
        subprocess.run([PY, f"{H}/wer.py", *dirs])
    subprocess.run([PY, f"{H}/summ.py", f"/workspace/results/{args.tag}"])
    print("AB DONE", flush=True)


if __name__ == "__main__":
    main()
