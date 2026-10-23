#!/bin/bash
# Start the supervisor (which keeps queue.py running) detached from any ssh session.
# Refuses if a supervisor is already running.
if pgrep -f "bench/supervisor[.]py" >/dev/null; then echo "supervisor already running"; exit 1; fi
cd /root/work
rm -f /root/work/bench/.planned_restart
setsid nohup /root/work/clientenv/bin/python -I /root/work/bench/supervisor.py \
  >> /workspace/results/bench/supervisor.log 2>&1 < /dev/null &
sleep 3
pgrep -af "bench/supervisor[.]py|bench/benchq[.]py"
