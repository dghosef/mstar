#!/bin/bash
# Text-only results back from the box (Mac disk is full: no media, no traces, logs <= 5 MB).
rsync -a --max-size=5m -e "ssh -F $HOME/runs/overnight/ssh_config" --include='*/' --include='*.md' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.txt' --include='*.log' --include='*.py' --include='*.sh' --include='*.yaml' --exclude='*' ovn-omni:/workspace/results/ $HOME/runs/overnight/results/omni/
