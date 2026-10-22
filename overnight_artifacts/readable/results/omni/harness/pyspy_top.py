#!/usr/bin/env python3
"""Summarize py-spy raw (collapsed) output: per thread, top frames by inclusive and self samples.

  python pyspy_top.py pyspy_<pid>.txt [N] [filter-substring]
"""
import collections
import sys

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
flt = sys.argv[3] if len(sys.argv) > 3 else None
incl = collections.defaultdict(collections.Counter)
self_ = collections.defaultdict(collections.Counter)
tot = collections.Counter()
for line in open(path):
    stack, _, n = line.rstrip().rpartition(" ")
    n = int(n)
    frames = stack.split(";")
    thread = frames[0]
    tot[thread] += n
    seen = set()
    for f in frames[1:]:
        key = f.split(" (")[0] + " " + f.split("/")[-1][:60]
        if key not in seen:
            incl[thread][key] += n
            seen.add(key)
    if len(frames) > 1:
        f = frames[-1]
        self_[thread][f.split(" (")[0] + " " + f.split("/")[-1][:60]] += n
for thread, t in tot.most_common(4):
    print(f"=== {thread[:80]}  samples={t}")
    for k, v in incl[thread].most_common(200):
        if flt and flt not in k:
            continue
        print(f"  incl {100*v/t:5.1f}%  {k}")
        top -= 1
        if top <= 0:
            break
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    print("  -- self --")
    for k, v in self_[thread].most_common(12):
        print(f"  self {100*v/t:5.1f}%  {k}")
