#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/queue16.sh" >/dev/null; do sleep 15; done
rsync -a /root/work/t_head3/ /root/work/t_q17bf/ && printf "model_kwargs:\n  codec_dtype: bfloat16\n" >> /root/work/t_q17bf/configs/qwen3tts_1p7b.yaml
python ab.py --arms head=/root/work/t_head3,bf16=/root/work/t_q17bf --rounds 2 --out /workspace/results/q17_bf16 -- --model qwen3_tts_1p7b --config configs/qwen3tts_1p7b.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 1,32,64 --save-audio 32
