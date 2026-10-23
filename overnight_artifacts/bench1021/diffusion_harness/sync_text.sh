#!/bin/bash
# Mac side: text-only results from node2 -> ~/runs/bench1021/results/diffusion/ (no images/videos/traces/npy)
rsync -az -e "ssh -F $HOME/runs/bench1021/ssh_config" --prune-empty-dirs \
  --include='*/' --include='*.md' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.log' \
  --include='*.txt' --include='*.cmd' --include='*.jpg' --exclude='*' \
  node2:/workspace/results/bench/ "$HOME/runs/bench1021/results/diffusion/"
