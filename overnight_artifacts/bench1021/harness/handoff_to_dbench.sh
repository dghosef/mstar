#!/bin/bash
# When node8's own queue (benchq) reports DONE, stop it and start node2's harness (dbench/dqueue)
# on the models moved here to even out the two nodes. Both runners write the same
# /workspace/results/bench/heartbeat.txt and failures.jsonl, so only one may run at a time.
until grep -q "model DONE" /workspace/results/bench/heartbeat.txt 2>/dev/null; do sleep 30; done
echo "benchq done: $(date -u)"
bash /root/work/bench/stop_runner.sh
rm -f /workspace/results/bench/DONE /root/work/dbench/.planned_restart /root/work/dbench/dqueue.pid
cd /root/work && setsid nohup /usr/bin/python3 /root/work/dbench/supervisor.py \
  >> /workspace/results/bench/supervisor.log 2>&1 < /dev/null &
echo "dbench supervisor started: $(date -u)"
