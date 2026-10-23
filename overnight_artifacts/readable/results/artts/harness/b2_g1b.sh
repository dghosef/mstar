#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
R=/workspace/results/b2_fish_w13
python bench_arm.py --tree /root/work/t_fnorm2 --label fnorm2_r1 --gpu 1 --port 8300 --model fish_s2_pro --config configs/fish_s2_pro.yaml --sample-rate 44100 --concs 1,32,64 --save-audio 32 --out $R
python bench_arm.py --tree /root/work/t_fnorm2 --label fnorm2_r2 --gpu 1 --port 8300 --model fish_s2_pro --config configs/fish_s2_pro.yaml --sample-rate 44100 --concs 1,32,64 --save-audio 32 --out $R
