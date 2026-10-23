#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/queue12.sh" >/dev/null; do sleep 15; done
python ab.py --arms fnorm=/root/work/t_q10,clonecap=/root/work/t_q11 --rounds 2 --out /workspace/results/qbase_ab3 -- --model qwen3_tts_base --config configs/qwen3tts_base.yaml --gpu 0 --port 8100 --language English --concs 1,16,32 --save-audio 16 --extra-file ref_audio=/root/work/ref_b64.txt --extra "ref_text=The quick brown fox jumps over the lazy dog."
