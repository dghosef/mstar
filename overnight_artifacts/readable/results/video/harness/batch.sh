#!/bin/bash
# batch.sh <batchname> then lines on stdin: "<label> <args...>"; runs sequentially, logs per job
B=$1; cd /root/work
while read -r label rest; do
  [ -z "$label" ] && continue
  mstar/.venv/bin/python vharness/ovn_video_bench.py run --out /workspace/results/video --python /root/work/mstar/.venv/bin/python --env HF_HOME=/root/work/hf --label $label $rest > /workspace/results/video/$label.log 2>&1
  grep "^RESULT" /workspace/results/video/$label.log >> /workspace/results/video/$B.results || echo "FAILED $label" >> /workspace/results/video/$B.results
done
echo BATCH_DONE >> /workspace/results/video/$B.results
