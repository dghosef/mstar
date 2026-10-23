#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
R=/workspace/results/b3_fish_fn3
C="--model fish_s2_pro --config configs/fish_s2_pro.yaml --sample-rate 44100 --concs 1,32,64 --save-audio 32 --out $R"
python bench_arm.py --tree /root/work/t_w13 --label w13_r1 --gpu 0 --port 8100 $C > /root/work/b3_w13.log 2>&1 &
python bench_arm.py --tree /root/work/t_fn3 --label fn3_r1 --gpu 1 --port 8300 $C > /root/work/b3_fn3.log 2>&1 &
wait
echo DONE
