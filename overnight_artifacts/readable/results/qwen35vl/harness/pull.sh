#!/bin/bash
rsync -a --max-size=5m --include='*/' --include='*.md' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.txt' --include='*.log' --include='*.py' --include='*.sh' --include='*.yaml' --exclude='*' -e "ssh -F $HOME/runs/overnight/ssh_config" ovn-qwen35vl:/workspace/results/ ~/runs/overnight/results/qwen35vl/
