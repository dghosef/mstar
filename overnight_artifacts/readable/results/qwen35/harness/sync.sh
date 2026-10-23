#!/bin/bash
# text-only results sync (PLAYBOOK UPDATE 22:35); skips per-request outputs and logs > 5 MB
rsync -a -e "ssh -F $HOME/runs/overnight/ssh_config" --max-size=5m --exclude='req_*.txt' --include='*/' --include='*.md' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.txt' --include='*.log' --include='*.py' --include='*.sh' --include='*.yaml' --exclude='*' ovn-qwen35:/workspace/results/ ~/runs/overnight/results/qwen35/
