#!/bin/bash
# gq.sh <gpu> <port> <arm-spec>... ; each spec tree|arm|size ; runs sequentially on one GPU
GPU=$1; PORT=$2; shift 2
for spec in "$@"; do IFS='|' read -r tree arm size <<< "$spec"
  echo "$(date +%H:%M:%S) >>> $arm"; /root/work/mstar/.venv/bin/python /root/guest-qwen35vl/gvlh/garm.py --tree $tree --arm $arm --size $size --gpu $GPU --port $PORT
done; echo "QUEUE DONE"
