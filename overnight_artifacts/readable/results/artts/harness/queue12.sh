#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/queue11.sh" >/dev/null; do sleep 15; done
python ab.py --arms head=/root/work/t_head2,fnorm=/root/work/t_q10 --rounds 3 --out /workspace/results/q06_ab12 -- --model qwen3_tts --config configs/qwen3tts.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 1,32,64 --save-audio 32
