#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
python ab.py --arms pfcs=/root/work/t_fish3,tf32=/root/work/t_fish4 --rounds 2 --out /workspace/results/fish_ab3 -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,16,32 --save-audio 16 --greedy do_sample=false
