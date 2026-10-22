#!/bin/bash
# GPU microbenches (GPU1, after the Fish A/Bs)
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf CUDA_VISIBLE_DEVICES=1
while pgrep -f "^/bin/bash /root/work/harness_artts/fishq_g1e.sh" >/dev/null; do sleep 15; done
R=/workspace/results/micro; mkdir -p $R
PYTHONPATH=/root/work/t_fish4 python fish_codec_micro.py --tf32 --bs 1,4,16 --widths 12,60,252 > $R/fish_tf32_nobench.txt 2>&1
PYTHONPATH=/root/work/t_fish4 python fish_codec_micro.py --tf32 --bs 1,4,16 --widths 12,60,252 --cudnn-bench > $R/fish_tf32_bench.txt 2>&1
PYTHONPATH=/root/work/t_fish4 python codec_micro.py --bs 1,4,16,32 --widths 1,12,53 > $R/qwen_nobench.txt 2>&1
PYTHONPATH=/root/work/t_fish4 python codec_micro.py --bs 1,4,16,32 --widths 1,12,53 --cudnn-bench > $R/qwen_bench.txt 2>&1
echo MICRO DONE
