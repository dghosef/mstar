"""Keeps benchq.py running until the queue is done.

  setsid nohup /root/work/clientenv/bin/python /root/work/bench/supervisor.py \
      > /workspace/results/bench/supervisor.log 2>&1 < /dev/null &

Runs benchq.py (its log: /workspace/results/bench/queue.log, appended); if it exits without
"QUEUE DONE", waits 60 s and runs it again (it resumes from its markers and results.jsonl).
benchq.py writes the heartbeat while it runs; between runs this script does. When the queue
is done it writes DONE to the heartbeat and exits.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = Path("/workspace/results/bench")
HEART = RES / "heartbeat.txt"
QLOG = RES / "queue.log"


def beat(msg):
    HEART.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} supervisor: {msg}\n")


def main():
    n = 0
    while True:
        n += 1
        beat(f"starting benchq.py (run {n})")
        with QLOG.open("a") as f:
            f.write(f"\n===== supervisor run {n} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            f.flush()
            rc = subprocess.run([sys.executable, "-I", str(HERE / "benchq.py")], stdout=f, stderr=subprocess.STDOUT,
                                cwd="/root/work").returncode
        tail = QLOG.read_text()[-4000:]
        if rc == 0 and "QUEUE DONE" in tail:
            beat("DONE")
            print("DONE", flush=True)
            return
        planned = HERE / ".planned_restart"
        if planned.exists():   # a deliberate restart (stop_runner.sh): not a failure
            planned.unlink()
            continue
        if rc == 3:            # benchq.py reloading itself (new harness code or a .reload flag)
            continue
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": "-", "arm": "-",
               "phase": "runner", "cell": "-", "error": "\n".join(tail.splitlines()[-20:]),
               "action_taken": f"benchq.py exited rc={rc}; supervisor restarts it in 60 s (run {n + 1})"}
        with (RES / "failures.jsonl").open("a") as f:
            f.write(__import__("json").dumps(rec) + "\n")
        for i in range(6):
            beat(f"benchq.py exited rc={rc} (run {n}); restarting in {60 - 10 * i}s")
            time.sleep(10)


if __name__ == "__main__":
    main()
