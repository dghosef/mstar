#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/queue7.sh" >/dev/null; do sleep 15; done
python ab.py --arms q6=/root/work/t_q6,q8x=/root/work/t_q8x --rounds 2 --out /workspace/results/q06_ab8 -- --model qwen3_tts --config configs/qwen3tts.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 32,64 --save-audio ""
