#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f prof_queue.sh >/dev/null || pgrep -f fish_queue3.sh >/dev/null; do sleep 15; done
python ab.py --arms q2b=/root/work/t_q2b,q5=/root/work/t_q5 --rounds 3 --out /workspace/results/q06_ab4 -- --model qwen3_tts --config configs/qwen3tts.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 1,32,64 --save-audio 32
