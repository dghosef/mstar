#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/fishq_sched.sh" >/dev/null; do sleep 15; done
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/work/t_head4 python fish_codec_dtype.py > /workspace/results/micro/fish_codec_dtype.txt 2>&1
echo MICRO2 DONE
