#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/fishq_g1d.sh" >/dev/null; do sleep 15; done
python ab.py --arms tf32=/root/work/t_fish4,dec64=/root/work/t_fish7 --rounds 2 --out /workspace/results/fish_ab6 -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,32,64 --save-audio 32
