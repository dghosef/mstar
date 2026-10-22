"""All NVTX ranges in [a,b] ms relative to the timeline origin used by nsys_timeline (fraction t_from)."""
import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); t_from=float(sys.argv[2]); a=float(sys.argv[3]); b=float(sys.argv[4])
mind = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
cols = [r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")]
txt = "COALESCE(text, (SELECT value FROM StringIds WHERE id=textId))" if "textId" in cols else "text"
nv = db.execute(f"SELECT start, end, {txt}, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchall()
lo = min(x for x, *_ in nv); hi = max(y for _, y, *_ in nv); t0 = lo + (hi - lo) * t_from
for s, e, t, tid in sorted(nv):
    rs, re_ = (s - t0) / 1e6, (e - t0) / 1e6
    if re_ >= a and rs <= b and (e - s) / 1e6 >= mind and t and "run_loop" not in t:
        print(f"{rs:9.2f} {re_:9.2f} dur={re_-rs:7.2f} tid={tid % 100000:6d} {t[:90]}")
