#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "wer_dirs.py" >/dev/null; do sleep 15; done
python ab.py --arms head=/root/work/t_head3,clonecap=/root/work/t_fish8 --rounds 2 --out /workspace/results/fish_clone_ab -- --model fish_s2_pro --config configs/fish_s2_pro.yaml --gpu 1 --port 8300 --sample-rate 44100 --concs 1,16,32 --save-audio 16 --extra-file ref_audio=/root/work/ref_b64.txt --extra "ref_text=The quick brown fox jumps over the lazy dog."
