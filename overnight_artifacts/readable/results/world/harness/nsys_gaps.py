"""GPU idle gaps in an nsys sqlite export: histogram and the kernels on either side of big gaps."""
import sqlite3
import sys
from collections import Counter

db = sqlite3.connect(sys.argv[1])
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 50e3  # ns
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = sorted(db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall())
merged = []
for s, e, n in rows:
    if merged and s <= merged[-1][1]:
        if e > merged[-1][1]:
            merged[-1][1] = e
            merged[-1][3] = n
    else:
        merged.append([s, e, n, n])
gaps = [(merged[i + 1][0] - merged[i][1], names.get(merged[i][3], "?")[:60], names.get(merged[i + 1][2], "?")[:60])
        for i in range(len(merged) - 1)]
tot = sum(g for g, _, _ in gaps)
big = [g for g in gaps if g[0] > thr]
print(f"gaps total {tot/1e6:.1f} ms over {len(gaps)}; > {thr/1e3:.0f}us: {len(big)} totalling {sum(g for g,_,_ in big)/1e6:.1f} ms")
for lo, hi in ((0, 5e3), (5e3, 20e3), (20e3, 100e3), (100e3, 1e6), (1e6, 1e12)):
    sel = [g for g, _, _ in gaps if lo <= g < hi]
    print(f"  {lo/1e3:7.0f}-{hi/1e3:7.0f}us: n={len(sel):6d} sum={sum(sel)/1e6:8.2f} ms")
c = Counter((a, b) for g, a, b in big)
for (a, b), k in c.most_common(12):
    s = sum(g for g, x, y in big if (x, y) == (a, b))
    print(f"{k:5d} {s/1e6:8.2f}ms  after {a}  ||  before {b}")
