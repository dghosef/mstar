"""Print node-level NVTX ranges (worker[...]node[...]) in time order with GPU kernel busy time inside each, for a window."""
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
t_from = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5  # fraction of trace
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
cols = [r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")]
txt = "COALESCE(text, (SELECT value FROM StringIds WHERE id=textId))" if "textId" in cols else "text"
nv = db.execute(f"SELECT start, end, {txt}, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchall()
k = sorted(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall())
lo = min(a for a, *_ in nv); hi = max(b for _, b, *_ in nv)
t0 = lo + (hi - lo) * t_from
sel = sorted([r for r in nv if r[0] >= t0 and r[2] and ("node[" in r[2] or "conductor._do_ingest" in r[2] or "dataworker" in r[2] or r[2].startswith("engine.forward"))])[:n]
import bisect
starts = [a for a, _ in k]
def busy(a, b):
    i = bisect.bisect_left(starts, a) - 1
    tot = 0
    for s, e in k[max(i, 0):]:
        if s > b: break
        tot += max(0, min(e, b) - max(s, a))
    return tot
prev = None
for a, b, t, tid in sel:
    print(f"{(a - t0)/1e6:9.2f}ms dur={(b - a)/1e6:8.2f}ms gpu={busy(a, b)/1e6:7.2f}ms tid={tid % 100000:6d} {t[:80]}")
