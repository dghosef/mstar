#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
python ab.py --arms head=/root/work/t_head4,cud=/root/work/t_cud --rounds 2 --out /workspace/results/b2_fish_cud -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,16,32,64 --save-audio 16
