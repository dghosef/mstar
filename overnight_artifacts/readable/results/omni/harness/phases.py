#!/usr/bin/env python3
"""phases.py server.log [worker] [min_bs] : the longest segment per worker with batch >= min_bs."""
import sys
sys.path.insert(0, "/root/work/base")
from benchmark.worker_phases.phases import parse_log, segment, render  # noqa: E402
segs = segment(parse_log(open(sys.argv[1]).read()), min_records=2)
want = sys.argv[2] if len(sys.argv) > 2 else None
min_bs = float(sys.argv[3]) if len(sys.argv) > 3 else 0
txt = render(segs)
blocks = [b for b in txt.split("\n\n") if b.strip().startswith("===")]
keep = []
for b in blocks:
    head = b.splitlines()[0]
    if want and want not in head:
        continue
    bs = float(head.split("batch=")[1].split()[0])
    if bs >= min_bs:
        keep.append((int(head.split("records=")[1].split()[0]), b))
for _, b in sorted(keep, reverse=True)[:1]:
    print("\n".join(l for l in b.splitlines() if not any(x in l for x in ("res.", "rust.", "stop_loops", "update_lru", "engine.pre_"))))
