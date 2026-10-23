#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f q_queue.sh >/dev/null; do sleep 15; done
PYTHONPATH=/root/work/t_fish3 CUDA_VISIBLE_DEVICES=0 python fish_snake_check.py > /workspace/results/fish_snake_check.txt 2>&1
python ab.py --arms pf=/root/work/t_fish1,pfc=/root/work/t_fish2,pfcs=/root/work/t_fish3 --rounds 2 --out /workspace/results/fish_ab2 -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 0 --port 8200 --sample-rate 44100 --concs 1,16,32 --save-audio 16 --greedy do_sample=false
