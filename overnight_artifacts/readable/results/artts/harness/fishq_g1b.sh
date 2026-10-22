#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/fishq_g1.sh" >/dev/null; do sleep 15; done
python bench_arm.py --tree /root/work/t_fish5 --label pfcd_r1 --out /workspace/results/fish_ab4 --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,16,32 --save-audio 16 --env TORCHINDUCTOR_CACHE_DIR=/root/work/inductor_cold_default
python ab.py --arms pfcs=/root/work/t_fish3,pfcd=/root/work/t_fish5 --rounds 1 --start-round 2 --out /workspace/results/fish_ab4 -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,16,32 --save-audio 16
