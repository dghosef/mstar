#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "wer_dirs.py" >/dev/null; do sleep 15; done
python ab.py --arms base=/root/work/base2,head=/root/work/t_head3 --rounds 2 --out /workspace/results/fish_final -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,16,32,64 --save-audio 16
