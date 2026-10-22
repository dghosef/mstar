#!/bin/bash
cd /root/guest-artts/harness
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf GUEST_PYLIB=/root/guest-artts/pylib GUEST_INDUCTOR=/root/guest-artts/inductor
R=/workspace/results/artts/fish_fnorm3
C="--model fish_s2_pro --config configs/fish_s2_pro.yaml --sample-rate 44100 --concs 1,32,64 --save-audio 32 --out $R"
python bench_arm.py --tree /root/guest-artts/w13 --label w13_r1 --gpu 0 --port 9100 $C > $R.w13_r1.log 2>&1 &
python bench_arm.py --tree /root/guest-artts/fnorm3 --label fnorm3_r1 --gpu 1 --port 9101 $C > $R.fnorm3_r1.log 2>&1 &
wait
python bench_arm.py --tree /root/guest-artts/fnorm3 --label fnorm3_r2 --gpu 0 --port 9100 $C > $R.fnorm3_r2.log 2>&1 &
python bench_arm.py --tree /root/guest-artts/w13 --label w13_r2 --gpu 1 --port 9101 $C > $R.w13_r2.log 2>&1 &
wait
echo AB DONE
