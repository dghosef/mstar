#!/usr/bin/env python3
"""Phase means over records whose in-flight bs is in [lo, hi), from a server.log.

    python phase_bs.py server.log 40 80 [phase ...]
Prints n-weighted means, plus gpu_exec/iter_total (GPU-stream busy share) and,
when the per-walk diag span exists, GPU time by walk.
"""
import collections
import re
import sys

ITER = sys.argv[1] == "--iter"
if ITER:
    sys.argv.pop(1)
log, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
want = sys.argv[4:] or ["iter_total", "gpu_exec", "worker.postprocess.event_sync", "worker.gpu_thread.prepare_inputs",
                        "worker.gpu_thread.exec", "worker.plan_thread.pre_plan", "worker.postprocess_batch",
                        "submit_spec", "worker.postprocess.check_stop", "engine.collect_outputs", "await_gpu"]
pat = re.compile(r"([\w.#]+): p50=([\d.]+)ms p95=([\d.]+)ms mean=([\d.]+)ms n=(\d+)")
tot = collections.defaultdict(lambda: [0.0, 0])
recs = 0
for line in open(log, errors="replace"):
    if "phase-timing" not in line:
        continue
    m = re.search(r"iter=(\d+) bs=([\d.]+)", line)
    key = float(m.group(1)) if (m and ITER) else (float(m.group(2)) if m else None)
    if key is None or not (lo <= key < hi):
        continue
    recs += 1
    for k, _, _, mean, n in pat.findall(line):
        tot[k][0] += float(mean) * int(n)
        tot[k][1] += int(n)
print(f"records={recs}")
for k in want:
    s, n = tot.get(k, (0, 0))
    if n:
        print(f"{k:40s} mean={s / n:8.3f} ms  n={n}")
it, gx = tot.get("iter_total", (0, 0)), tot.get("gpu_exec", (0, 0))
if it[0]:
    print(f"gpu_exec/iter_total sum = {gx[0] / it[0]:.2f}")
for k in sorted(tot):
    if k.startswith("rows."):
        s, n = tot[k]
        print(f"{k:40s} mean_rows={s / n:6.2f} steps={n}")
walks = collections.defaultdict(float)
for k, (s, n) in tot.items():
    if k.startswith("gpu_exec."):
        walks[k.split(".")[1]] += s
for w, s in walks.items():
    print(f"GPU {w:20s} {s / 1000:8.2f} s")
