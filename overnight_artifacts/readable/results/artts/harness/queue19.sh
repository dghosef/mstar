#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
python ab.py --arms head=/root/work/t_head4,swi=/root/work/t_q13 --rounds 3 --out /workspace/results/q06_ab19 -- --model qwen3_tts --config configs/qwen3tts.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 1,32,64 --save-audio ""
