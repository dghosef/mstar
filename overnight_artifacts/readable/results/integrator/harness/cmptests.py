import json
import sys

b, c = (json.load(open(f"{p}/tests.json")) for p in sys.argv[1:3])
for k in c:
    nb, nc = set(b[k]["failed"]), set(c[k]["failed"])
    print(f"{k}: total {b[k]['total']} -> {c[k]['total']}, failed {len(nb)} -> {len(nc)}, "
          f"NEW: {sorted(nc - nb)}, fixed: {sorted(nb - nc)}")
