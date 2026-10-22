#!/bin/bash
# push my worktree to the box's change tree
C="-F $HOME/runs/overnight/ssh_config"
rsync -a -e "ssh $C" --exclude .venv --exclude target --exclude __pycache__ --exclude '*.so' --exclude .git /Users/josephtan/mstar-ovn-qwen35/ ovn-qwen35:/root/work/mstar/
