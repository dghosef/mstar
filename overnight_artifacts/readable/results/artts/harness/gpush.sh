#!/bin/bash
# gpush.sh <box> <tree-name> : worktree -> /root/guest-artts/<tree-name>; harness -> /root/guest-artts/harness
B=${1:?box}; T=${2:?tree}
rsync -a --exclude .venv --exclude target --exclude __pycache__ --exclude .git --exclude '*.egg-info' -e "ssh -F $HOME/runs/overnight/ssh_config" /Users/josephtan/mstar-ovn-artts/ $B:/root/guest-artts/$T/
rsync -a -e "ssh -F $HOME/runs/overnight/ssh_config" ~/runs/overnight/results/artts/harness/ $B:/root/guest-artts/harness/
