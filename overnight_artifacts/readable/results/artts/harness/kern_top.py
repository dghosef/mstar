#!/usr/bin/env python3
"""Top kernels from an nsys cuda_gpu_kern_sum csv: kern_top.py file.csv [n]"""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
tot = sum(float(r["Total Time (ns)"]) for r in rows)
print(f"total GPU kernel ms {tot / 1e6:.1f}  instances {sum(int(r['Instances']) for r in rows)}")
for r in rows[:n]:
    print(f"{float(r['Time (%)']):5.1f}% {int(r['Instances']):8d} avg {float(r['Avg (ns)']) / 1000:7.1f}us  {r['Name'][:120]}")
