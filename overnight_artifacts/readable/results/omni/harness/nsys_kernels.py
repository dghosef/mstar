#!/usr/bin/env python3
"""Kernel time by name per process from an nsys sqlite export.

  python nsys_kernels.py report.sqlite [topN]
"""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
q = """
SELECT k.globalPid / 16777216 % 16777216 AS pid, s.value AS name, COUNT(*) AS n,
       SUM(k.end - k.start) AS tot, MIN(k.start) AS t0, MAX(k.end) AS t1, k.deviceId AS dev
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = COALESCE(k.shortName, k.demangledName)
GROUP BY pid, name ORDER BY pid, tot DESC
"""
rows = db.execute(q).fetchall()
by_pid = {}
for pid, name, n, tot, t0, t1, dev in rows:
    by_pid.setdefault(pid, []).append((name, n, tot, t0, t1, dev))
for pid, ks in by_pid.items():
    total = sum(k[2] for k in ks)
    t0 = min(k[3] for k in ks)
    t1 = max(k[4] for k in ks)
    # union of kernel intervals = busy time
    iv = db.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid / 16777216 % 16777216 = ? ORDER BY start",
        (pid,)).fetchall()
    busy, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    nk = sum(k[1] for k in ks)
    print(f"=== pid {pid} dev {ks[0][5]}: kernels={nk} kernel_time={total/1e6:.1f}ms "
          f"busy={busy/1e6:.1f}ms span={(t1-t0)/1e6:.1f}ms util={busy/max(t1-t0,1):.2%}")
    for name, n, tot, *_ in ks[:top]:
        print(f"  {tot/1e6:9.2f}ms {100*tot/total:5.1f}% n={n:7d} avg={tot/n/1e3:7.2f}us  {name[:110]}")
