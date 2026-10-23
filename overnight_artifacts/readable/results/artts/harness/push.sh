#!/bin/bash
# push my worktree to a tree on the box: push.sh <name>  -> /root/work/<name>
T=${1:?tree name}
rsync -a --exclude .venv --exclude target --exclude __pycache__ --exclude .git --exclude '*.egg-info' -e "ssh -F $HOME/runs/overnight/ssh_config" /Users/josephtan/mstar-ovn-artts/ ovn-artts:/root/work/$T/
rsync -a -e "ssh -F $HOME/runs/overnight/ssh_config" ~/runs/overnight/results/artts/harness/ ovn-artts:/root/work/harness_artts/
