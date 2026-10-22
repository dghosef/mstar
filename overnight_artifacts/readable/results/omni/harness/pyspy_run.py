#!/usr/bin/env python3
"""Run bench.py for one arm and py-spy the GPU-<gpu> worker during the named workload.

  python pyspy_run.py --tree /root/work/mstar --tag prof1 --workloads t2s_c32 --gpu 0 \
      --delay 25 --seconds 20 [--env TORCH_LOGS=recompiles]
"""
import argparse
import os
import subprocess
import sys
import time

H = "/root/work/harness_omni"
PY = "/root/work/mstar/.venv/bin/python"


def gpu_pids(gpu: int) -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    bus = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return [int(line.split(",")[0]) for line in out.splitlines() if line.split(",")[1].strip() == bus]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", default="prof")
    ap.add_argument("--workloads", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--delay", type=float, default=25)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--gil", action="store_true")
    args = ap.parse_args()
    out = f"/workspace/results/{args.tag}/{args.arm}"
    os.makedirs(out, exist_ok=True)
    cmd = [PY, f"{H}/bench.py", "--tree", args.tree, "--arm", args.arm, "--tag", args.tag,
           "--workloads", args.workloads, "--phase-timing", "100"]
    for e in args.env:
        cmd += ["--env", e]
    log = open(f"{out}/../{args.arm}_bench.log", "w")
    bench = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    # wait until the client of the first workload has started
    while not os.path.exists(f"{out}/{args.workloads.split(',')[0]}/bench.log"):
        if bench.poll() is not None:
            sys.exit("bench exited early")
        time.sleep(2)
    time.sleep(args.delay)
    pids = gpu_pids(args.gpu)
    print("gpu pids", pids, flush=True)
    for pid in pids:
        subprocess.run(["/root/work/mstar/.venv/bin/py-spy", "record", "-p", str(pid), "-r", "100",
                        "-d", str(int(args.seconds)), "-f", "raw", "-o", f"{out}/pyspy_{pid}.txt",
                        "--threads", "--nonblocking", *(["--gil"] if args.gil else [])], capture_output=True)
    bench.wait()
    print("PYSPY DONE", flush=True)


if __name__ == "__main__":
    main()
