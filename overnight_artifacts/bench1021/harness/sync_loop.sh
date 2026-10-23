#!/bin/bash
# Mac side: copy text results off node8 every 10 min (md/json/jsonl/csv/log/txt/cmd/html; no media or traces).
while true; do
  rsync -az --include '*/' --include '*.md' --include '*.json' --include '*.jsonl' --include '*.csv' \
    --include '*.log' --include '*.txt' --include '*.cmd' --include '*.html' --exclude '*' \
    -e "ssh -F $HOME/runs/bench1021/ssh_config" node8:/workspace/results/bench/ "$HOME/runs/bench1021/results/" \
    && date "+%H:%M synced" >> "$HOME/runs/bench1021/results/.sync.log"
  sleep 600
done
