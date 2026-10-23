#!/bin/bash
# Mac side: push the diffusion harness to node2:/root/work/dbench
set -euo pipefail
rsync -az -e "ssh -F $HOME/runs/bench1021/ssh_config" "$HOME/runs/bench1021/diffusion_harness/" node2:/root/work/dbench/
