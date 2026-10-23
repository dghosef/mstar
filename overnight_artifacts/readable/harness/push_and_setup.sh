#!/bin/bash
# push_and_setup.sh <alias> [tree_dir] [src_tree]
#   rsync a tree + harness to the box and start setup (own venv inside tree_dir) in the background.
#   Guests on a shared box: push_and_setup.sh ovn-tts /root/guest-video/mstar /Users/josephtan/mstar-ovn-video
set -e
H=$1; D=${2:-/root/work/mstar}; S=${3:-/Users/josephtan/mstar-ovn}; C="-F $HOME/runs/overnight/ssh_config"
ssh $C $H "mkdir -p $D /root/work"
rsync -a -e "ssh $C" --exclude .venv --exclude target --exclude __pycache__ --exclude '*.so' "$S"/ $H:$D/
rsync -a -e "ssh $C" $HOME/runs/overnight/harness/ $H:/root/work/harness/
L=$(dirname $D)/setup.log
ssh $C $H "nohup setsid bash /root/work/harness/setup_box.sh $D > $L 2>&1 < /dev/null & echo started, log $L"
