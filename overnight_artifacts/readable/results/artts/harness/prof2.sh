#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/queue18.sh" >/dev/null; do sleep 15; done
python bench_arm.py --tree /root/work/t_head3 --label head_spy --out /workspace/results/q06_prof2 --model qwen3_tts --config configs/qwen3tts.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 64 --save-audio "" --pyspy-cell 64 --pyspy-secs 12
