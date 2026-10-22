#!/bin/bash
# Push the omni worktree to the box's /root/work/mstar (the change arm).
rsync -a --delete -e "ssh -F $HOME/runs/overnight/ssh_config" \
  --exclude .git --exclude .venv --exclude target --exclude __pycache__ --exclude '*.egg-info' \
  --exclude mstar-benchmark-cache --exclude .bench_outs \
  /Users/josephtan/mstar-ovn-omni/ ovn-omni:/root/work/mstar/
