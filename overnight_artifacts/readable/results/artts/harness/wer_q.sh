#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/micro_q.sh" >/dev/null; do sleep 15; done
A=/workspace/results/q06_ab7; B=/workspace/results/q06_ab6
python wer_dirs.py --sentences sentences_260.txt --device cuda:1 $A/q5_r1/c32_wav $A/q5_r2/c32_wav $A/q6_r1/c32_wav $A/q6_r2/c32_wav $B/single_r1/c32_wav $B/splitmps_r1/c32_wav > /root/work/wer_q.log 2>&1
echo WER DONE >> /root/work/wer_q.log
