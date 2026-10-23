#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/fishq_g1c.sh" >/dev/null; do sleep 15; done
rsync -a /root/work/t_fish4/ /root/work/t_fish6/ && printf "  chunk_frames: 64\n" >> /root/work/t_fish6/configs/fish_s2_pro.yaml
tail -3 /root/work/t_fish6/configs/fish_s2_pro.yaml
python ab.py --arms tf32=/root/work/t_fish4,ch64=/root/work/t_fish6 --rounds 2 --out /workspace/results/fish_ab5 -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,32,64 --save-audio 32 --greedy do_sample=false
