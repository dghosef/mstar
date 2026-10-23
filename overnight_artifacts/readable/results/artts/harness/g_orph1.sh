#!/bin/bash
cd /root/guest-artts/harness
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf GUEST_PYLIB=/root/guest-artts/pylib GUEST_INDUCTOR=/root/guest-artts/inductor
R=/workspace/results/artts/orpheus
python bench_arm.py --tree /root/guest-artts/obase --label base_r1 --gpu 0,1 --port 9100 --model orpheus --config configs/orpheus.yaml --concs 1,16,32,64 --save-audio 32 --greedy temperature=0 --out $R
echo DONE1
