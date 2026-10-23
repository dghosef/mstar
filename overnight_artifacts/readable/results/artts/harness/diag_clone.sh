#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
rsync -a /root/work/t_head2/ /root/work/d_head2/ && python3 diagify.py /root/work/d_head2
python bench_arm.py --tree /root/work/d_head2 --label diag_clone --out /workspace/results/qbase_diag --model qwen3_tts_base --config configs/qwen3tts_base.yaml --gpu 1 --port 8300 --language English --concs 1,32 --save-audio "" --extra-file ref_audio=/root/work/ref_b64.txt --extra "ref_text=The quick brown fox jumps over the lazy dog." --period 50
