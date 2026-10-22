#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/queue8.sh" >/dev/null; do sleep 15; done
rsync -a /root/work/t_head/ /root/work/t_q9/ && printf "model_kwargs:\n  codec_dtype: bfloat16\n" >> /root/work/t_q9/configs/qwen3tts.yaml
python ab.py --arms head=/root/work/t_head,bf16=/root/work/t_q9 --rounds 3 --out /workspace/results/q06_ab9 -- --model qwen3_tts --config configs/qwen3tts.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 1,32,64 --save-audio 32
