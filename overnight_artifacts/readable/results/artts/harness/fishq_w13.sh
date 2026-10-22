#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
python ab.py --arms head=/root/work/t_head3,w13=/root/work/t_fish9 --rounds 2 --out /workspace/results/fish_w13 -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,32,64 --save-audio 32
