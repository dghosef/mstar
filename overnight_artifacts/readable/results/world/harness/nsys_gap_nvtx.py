"""For the largest GPU idle gaps, list the NVTX ranges (any thread) overlapping each gap."""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = sorted(db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall())
merged = []
for s, e, n in rows:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
gaps = sorted(((merged[i + 1][0] - merged[i][1], merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)), reverse=True)
cols = [r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")]
txt = "COALESCE(text, (SELECT value FROM StringIds WHERE id=textId))" if "textId" in cols else "text"
nv = db.execute(f"SELECT start, end, {txt}, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchall()
PRE = float(sys.argv[5]) if len(sys.argv) > 5 else 0
MIN = float(sys.argv[6]) if len(sys.argv) > 6 else 0
lo = float(sys.argv[3]) if len(sys.argv) > 3 else 0
hi = float(sys.argv[4]) if len(sys.argv) > 4 else 1e18
gaps = [x for x in gaps if lo <= x[0] <= hi]
for g, s, e in gaps[:k]:
    print(f"--- gap {g/1e3:.0f}us")
    for a, b, t, tid in sorted(nv):
        if a < e and b > s - PRE and (b - a) < 50e6 and (b - a) > MIN:
            print(f"   tid={tid % 100000:6d} [{(a - s)/1e3:+8.0f}, {(b - s)/1e3:+8.0f}]us {str(t)[:90]}")
