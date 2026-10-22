#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/queue13.sh" >/dev/null; do sleep 15; done
R=/workspace/results/qbase_ab3
python wer_dirs.py --sentences sentences_260.txt --device cuda:0 $R/fnorm_r1/c16_wav $R/fnorm_r2/c16_wav $R/clonecap_r1/c16_wav $R/clonecap_r2/c16_wav > /root/work/wer_clone.log 2>&1
python ab.py --arms base=/root/work/base2,head=/root/work/t_head3 --rounds 2 --out /workspace/results/q06_final -- --model qwen3_tts --config configs/qwen3tts.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 1,16,32,64 --save-audio 16
python ab.py --arms base=/root/work/base2,head=/root/work/t_head3 --rounds 2 --out /workspace/results/q17_final -- --model qwen3_tts_1p7b --config configs/qwen3tts_1p7b.yaml --gpu 0 --port 8100 --voice vivian --language English --concs 1,16,32,64 --save-audio 16
