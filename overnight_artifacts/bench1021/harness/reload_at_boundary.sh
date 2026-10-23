#!/bin/bash
# One-off: restart the runner on new harness code at the next model boundary of the
# running (older) runner, i.e. as soon as the heartbeat shows model $1 in its setup phase.
#   setsid nohup bash reload_at_boundary.sh 3/ > /workspace/results/bench/reload.log 2>&1 &
POS=${1:-3/}
until grep -q "model ${POS}[0-9]* [^ ]* phase=setup" /workspace/results/bench/heartbeat.txt 2>/dev/null; do sleep 15; done
echo "boundary reached: $(cat /workspace/results/bench/heartbeat.txt)"
bash /root/work/bench/stop_runner.sh
bash /root/work/bench/start_runner.sh
echo "restarted $(date -u)"
