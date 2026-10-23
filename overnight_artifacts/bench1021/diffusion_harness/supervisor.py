#!/usr/bin/env python3
"""Keeps dqueue.py running (/root/work/dbench/supervisor.py).

  setsid nohup /usr/bin/python3 /root/work/dbench/supervisor.py >> /workspace/results/bench/supervisor.log 2>&1 &

dqueue.py never exits on its own (when the queue is empty it writes RESULTS/DONE and polls queue.json);
this reruns it whenever it dies (30 s pause, at most 40 restarts). It resumes from its stage markers and
completed cells. While it is down the heartbeat says so.

A runner exit is logged to failures.jsonl unless it was planned: touch /root/work/dbench/.planned_restart
before killing dqueue.py (or this supervisor) on purpose. If a dqueue.py is already running when this
starts (a supervisor restart), it waits for that one instead of starting a second.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path("/workspace/results/bench")
HB = RESULTS / "heartbeat.txt"
PLANNED = HERE / ".planned_restart"
PIDFILE = HERE / "dqueue.pid"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def running_dqueue():
    try:
        pid = int(PIDFILE.read_text())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        return pid if "dqueue.py" in cmd else None
    except (OSError, ValueError):
        return None


def log_exit(rc, attempt):
    if PLANNED.exists():
        PLANNED.unlink(missing_ok=True)
        print(f"{now()} planned restart (rc={rc}); not logged as a failure", flush=True)
        return
    try:
        tail = "\n".join((RESULTS / "queue.log").read_text(errors="replace").splitlines()[-20:])
    except OSError:
        tail = ""
    with open(RESULTS / "failures.jsonl", "a") as f:
        f.write(json.dumps({"time": now(), "model": "-", "arm": "-", "phase": "runner", "cell": "-",
                            "error": f"dqueue.py exited rc={rc}\n{tail}",
                            "action_taken": f"supervisor restarts dqueue.py in 30 s (run {attempt + 1})"}) + "\n")


def crash_signature():
    """(entry key, last error line) of the latest run in queue.log."""
    try:
        lines = (RESULTS / "queue.log").read_text(errors="replace").splitlines()[-400:]
    except OSError:
        return None, None
    key = None
    for ln in lines:
        if "==== queue " in ln:
            parts = ln.split("==== queue ", 1)[1].split()
            key = parts[1] if len(parts) > 1 else None
    errs = [ln for ln in lines if ln.startswith(("Traceback", "FileNotFoundError", "RuntimeError", "ValueError",
                                                 "KeyError", "TypeError", "AttributeError", "ImportError",
                                                 "ModuleNotFoundError", "OSError", "NameError", "Exception"))
            or "Error:" in ln[:60]]
    return key, (errs[-1][:300] if errs else None)


def main():
    pid = running_dqueue()
    if pid:
        print(f"{now()} dqueue.py already running (pid {pid}); waiting for it", flush=True)
        while running_dqueue():
            time.sleep(10)
        log_exit("unknown (adopted run)", 0)
        time.sleep(30)
    history = []
    for attempt in range(1, 41):
        print(f"{now()} starting dqueue.py (run {attempt})", flush=True)
        with open(RESULTS / "queue.log", "a") as f:
            p = subprocess.Popen([sys.executable, str(HERE / "dqueue.py")], stdout=f, stderr=subprocess.STDOUT,
                                 cwd="/tmp")
            PIDFILE.write_text(str(p.pid))
            rc = p.wait()
        print(f"{now()} dqueue.py exited rc={rc}", flush=True)
        planned = PLANNED.exists()
        log_exit(rc, attempt)
        sig = crash_signature()
        if not planned:
            history.append(sig)
            if len(history) >= 3 and history[-1] == history[-2] == history[-3] and sig[0]:
                stages = RESULTS / sig[0] / "stages"
                stages.mkdir(parents=True, exist_ok=True)
                (stages / "q_skipped.done").touch()
                with open(RESULTS / "failures.jsonl", "a") as f:
                    f.write(json.dumps({"time": now(), "model": sig[0], "arm": "-", "phase": "runner", "cell": "-",
                                        "error": f"3 identical runner crashes: {sig[1]}",
                                        "action_taken": "crash loop: entry marked failed (skipped); "
                                                        "queue moves to the next entry"}) + "\n")
                history.clear()
        else:
            history.clear()
        HB.write_text(f"{now()} | supervisor: dqueue.py exited rc={rc}, restarting in 30 s (run {attempt})\n")
        time.sleep(30)
    HB.write_text(f"{now()} | supervisor: gave up after 40 restarts\n")


if __name__ == "__main__":
    os.chdir("/tmp")
    main()
