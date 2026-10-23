#!/bin/bash
# Stop the supervisor, the runner and everything in their session (setup steps, clients),
# then the benchmark server. Writes no failures.jsonl line (the supervisor goes first).
touch /root/work/bench/.planned_restart   # so neither supervisor logs this as a failure
SUP=$(pgrep -f "bench/supervisor[.]py" | head -1)
if [ -n "$SUP" ]; then
  SID=$(ps -o sid= -p "$SUP" | tr -d " ")
  kill -STOP "$SUP" 2>/dev/null
  for p in $(ps -e -o pid=,sid= | awk -v s="$SID" '$2 == s {print $1}'); do [ "$p" != "$$" ] && kill -TERM "$p" 2>/dev/null; done
  sleep 3
  for p in $(ps -e -o pid=,sid= | awk -v s="$SID" '$2 == s {print $1}'); do [ "$p" != "$$" ] && kill -KILL "$p" 2>/dev/null; done
fi
/root/work/bench/server.sh stop bench >/dev/null 2>&1
pgrep -af "bench/supervisor[.]py|bench/benchq[.]py" || echo "runner stopped"
