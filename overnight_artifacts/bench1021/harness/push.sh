#!/bin/bash
# Mac side: push both trees and the harness to a box, then provision it.
#   bash push.sh node8
set -euo pipefail
H=${1:?host}
SSH="ssh -F $HOME/runs/bench1021/ssh_config"
PR=${PR_TREE:-/Users/josephtan/mstar-ovn-all}
MAIN=${MAIN_TREE:-/Users/josephtan/mstar-bench1021-main}
$SSH $H 'mkdir -p /root/work/bench /root/work/mstar /root/work/mstar-main /workspace/results/bench'
for pair in "$PR:mstar" "$MAIN:mstar-main"; do
  src=${pair%%:*}; dst=${pair##*:}
  sha=$(git -C "$src" rev-parse HEAD)
  rsync -az --delete --exclude .git --exclude .venv --exclude target --exclude overnight_artifacts \
        --exclude .setup_done --exclude COMMIT.txt -e "$SSH" "$src/" "$H:/root/work/$dst/"
  $SSH $H "echo '$(git -C "$src" rev-parse --abbrev-ref HEAD) $sha' > /root/work/$dst/COMMIT.txt"
done
rsync -az -e "$SSH" "$HOME/runs/bench1021/harness/" "$H:/root/work/bench/"
echo pushed
