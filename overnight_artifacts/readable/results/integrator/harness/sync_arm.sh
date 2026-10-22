#!/bin/bash
# sync_arm.sh <local worktree> <arm name> [box] [arms dir]: rsync a local tree to <arms dir>/<arm> on the box, stamping .arm_rev
set -euo pipefail
WT=$1; ARM=$2; BOX=${3:-ovn-omni}; ARMS=${4:-/root/guest-integrator/arms}; C="-F $HOME/runs/overnight/ssh_config"
REV="$(git -C "$WT" rev-parse --short HEAD) $(git -C "$WT" rev-parse --abbrev-ref HEAD) dirty=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
ssh $C $BOX "mkdir -p $ARMS/$ARM"
rsync -a --delete -e "ssh $C" --exclude .git --exclude .venv --exclude target --exclude __pycache__ --exclude '*.so' \
  --exclude 'rust/target' --exclude '.arm_rev' "$WT/" $BOX:$ARMS/$ARM/
ssh $C $BOX "echo '$REV' > $ARMS/$ARM/.arm_rev; find $ARMS/$ARM -name __pycache__ -prune -exec rm -rf {} +; cat $ARMS/$ARM/.arm_rev"
