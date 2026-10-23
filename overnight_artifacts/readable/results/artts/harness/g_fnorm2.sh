#!/bin/bash
cd /root/guest-artts/harness
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf GUEST_PYLIB=/root/guest-artts/pylib GUEST_INDUCTOR=/root/guest-artts/inductor
R=/workspace/results/artts/fish_fnorm2
C="--model fish_s2_pro --config configs/fish_s2_pro.yaml --sample-rate 44100 --concs 1,32,64 --save-audio 32 --out $R"
python bench_arm.py --tree /root/guest-artts/fnorm2 --label fnorm2_r1 --gpu 0 --port 9100 $C
python bench_arm.py --tree /root/guest-artts/fnorm2 --label fnorm2_r2 --gpu 0 --port 9100 $C
echo AB DONE
