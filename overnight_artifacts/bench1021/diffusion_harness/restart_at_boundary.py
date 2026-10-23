#!/usr/bin/env python3
"""One-shot: wait for the running dqueue.py to finish its current phase (a new q_<phase>.done marker),
then make the planned restart (touch .planned_restart, SIGTERM dqueue.py; the supervisor restarts it
in 30 s and it resumes from its markers). Used once to load dqueue.py's module dispatch + .reload flag;
later restarts just touch /root/work/dbench/.reload.

  setsid nohup nice -n 19 python3 restart_at_boundary.py "<note>" >> /workspace/results/bench/restart_watch.log 2>&1 &
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path("/root/work/dbench")
RESULTS = Path("/workspace/results/bench")


def markers():
    return {str(p): p.stat().st_mtime for p in RESULTS.glob("*/stages/q_*.done")}


def main():
    note = sys.argv[1] if len(sys.argv) > 1 else "planned restart at a phase boundary"
    pid = int((HERE / "dqueue.pid").read_text())
    base = markers()
    t0 = time.time()
    print(time.strftime("%H:%M:%S"), f"watching for a phase boundary (dqueue pid {pid})", flush=True)
    while True:
        cur = markers()
        new = [p for p, m in cur.items() if p not in base or m > base[p]]
        if new:
            break
        try:
            os.kill(pid, 0)
        except OSError:
            print(time.strftime("%H:%M:%S"), "dqueue.py already gone; nothing to do", flush=True)
            return
        time.sleep(0.5)
    (HERE / ".planned_restart").touch()
    os.kill(pid, signal.SIGTERM)
    with (RESULTS / "planned_restarts.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                            "note": f"{note} (boundary after {new[0]}, waited {time.time() - t0:.0f}s)"}) + "\n")
    print(time.strftime("%H:%M:%S"), f"restarted at boundary after {new}", flush=True)


if __name__ == "__main__":
    main()
