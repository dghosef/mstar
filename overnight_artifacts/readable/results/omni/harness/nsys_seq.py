#!/usr/bin/env python3
"""Print a kernel sequence window for one pid: python nsys_seq.py db pid start_index count"""
import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); pid = int(sys.argv[2]); s = int(sys.argv[3]); n = int(sys.argv[4])
rows = db.execute("""SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX FROM CUPTI_ACTIVITY_KIND_KERNEL k
 JOIN StringIds s ON s.id = COALESCE(k.shortName, k.demangledName)
 WHERE k.globalPid / 16777216 % 16777216 = ? ORDER BY k.start LIMIT ? OFFSET ?""", (pid, n, s)).fetchall()
prev = None
for st, en, name, gx, gy, gz, bx in rows:
    gap = (st - prev) / 1e3 if prev else 0
    print(f"{(en-st)/1e3:7.2f}us gap={gap:7.2f} grid=({gx},{gy},{gz}) blk={bx} {name[:90]}")
    prev = en
