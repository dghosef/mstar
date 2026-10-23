#!/usr/bin/env python3
"""Summarize a py-spy raw (collapsed) profile: top inclusive and self frames.

    python pyspy_top.py prof.txt [--filter substr] [--n 40]
"""
import argparse
import collections
import re

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--filter", default=None, help="only stacks containing this substring")
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()

incl = collections.Counter()
selfc = collections.Counter()
threads = collections.Counter()
total = 0
for line in open(args.path, errors="replace"):
    line = line.rstrip()
    if not line:
        continue
    stack, _, count = line.rpartition(" ")
    try:
        count = int(count)
    except ValueError:
        continue
    if args.filter and args.filter not in stack:
        continue
    frames = stack.split(";")
    total += count
    threads[frames[0]] += count
    names = []
    for f in frames:
        m = re.match(r"(.*?) \((.*?):\d+\)", f)
        names.append(f"{m.group(1)} ({m.group(2).split('/')[-1]})" if m else f)
    for n in set(names):
        incl[n] += count
    selfc[names[-1]] += count
print(f"total samples {total}")
print("threads:", threads.most_common(8))
print("\n== inclusive")
for n, c in incl.most_common(args.n):
    print(f"{100 * c / total:5.1f}%  {n}")
print("\n== self")
for n, c in selfc.most_common(args.n):
    print(f"{100 * c / total:5.1f}%  {n}")
