#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/fishq_g1b.sh" >/dev/null; do sleep 15; done
R=/workspace/results/fish_ab3
python wer_dirs.py --sentences sentences_260.txt --device cuda:1 $R/pfcs_r1/c16_wav $R/pfcs_r2/c16_wav $R/tf32_r1/c16_wav $R/tf32_r2/c16_wav > /root/work/wer_fish_ab3.log 2>&1
rsync -a /root/work/t_fish4/ /root/work/d_fish4/ && python3 diagify.py /root/work/d_fish4
python bench_arm.py --tree /root/work/d_fish4 --label diag_tf32 --out /workspace/results/fish_diag2 --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 32,64 --save-audio "" --period 50
