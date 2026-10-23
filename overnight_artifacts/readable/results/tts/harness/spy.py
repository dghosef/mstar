#!/usr/bin/env python3
"""py-spy a running mstar server's processes while a load runs.

    python spy.py --port 8100 --seconds 15 --out DIR [--match worker]

Finds the server's process tree (listener on --port and its descendants), records each
python process for --seconds at 100 Hz in raw (collapsed) format, and prints the top
self/inclusive frames per process whose stacks contain --match.
"""
import argparse
import collections
import os
import subprocess
import time
from pathlib import Path

PYSPY = "/root/work/mstar/.venv/bin/py-spy"


def tree(root: int) -> list[int]:
    out = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    kids = collections.defaultdict(list)
    for line in out.splitlines():
        p, pp = map(int, line.split())
        kids[pp].append(p)
    res, stack = [], [root]
    while stack:
        p = stack.pop()
        res.append(p)
        stack += kids[p]
    return res


def listener(port: int) -> int:
    out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True).stdout
    return int(out.split("pid=")[1].split(",")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", default="")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pids = [p for p in tree(listener(a.port)) if "python" in Path(f"/proc/{p}/exe").resolve().name]
    procs = []
    for p in pids:
        f = out / f"pyspy_{p}.txt"
        procs.append((p, f, subprocess.Popen([PYSPY, "record", "-p", str(p), "-d", str(a.seconds), "-r", "100",
                                              "--format", "raw", "-o", str(f), "--nonblocking"],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))
    for _, _, pr in procs:
        pr.wait()
    for p, f, _ in procs:
        if not f.exists():
            continue
        selfc, incl, total = collections.Counter(), collections.Counter(), 0
        for line in f.read_text().splitlines():
            stack, _, n = line.rpartition(" ")
            n = int(n)
            frames = stack.split(";")
            if a.match and not any(a.match in fr for fr in frames):
                continue
            total += n
            selfc[frames[-1]] += n
            for fr in set(frames):
                incl[fr] += n
        if not total:
            continue
        cmd = open(f"/proc/{p}/cmdline").read().replace("\0", " ")[:80] if os.path.exists(f"/proc/{p}/cmdline") else ""
        print(f"=== pid {p} samples {total} {cmd}")
        print("-- top self")
        for fr, n in selfc.most_common(a.top // 2):
            print(f"{100*n/total:6.1f}% {fr[:160]}")
        print("-- top inclusive")
        for fr, n in incl.most_common(a.top):
            print(f"{100*n/total:6.1f}% {fr[:160]}")


if __name__ == "__main__":
    time.sleep(0)
    main()
