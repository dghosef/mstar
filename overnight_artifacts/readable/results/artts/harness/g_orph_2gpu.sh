#!/bin/bash
cd /root/guest-artts/harness
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf GUEST_PYLIB=/root/guest-artts/pylib GUEST_INDUCTOR=/root/guest-artts/inductor
R=/workspace/results/artts/orpheus
while pgrep -f "g_orph_seq.sh" >/dev/null; do sleep 10; done
python bench_arm.py --tree /root/guest-artts/ofinal --label final2g_r1 --gpu 0,1 --port 9100 --model orpheus --config configs/orpheus.yaml --concs 1,16,32,64 --save-audio 32 --out $R > $R.final2g_r1.log 2>&1
python bench_arm.py --tree /root/guest-artts/obase --label base2g_r2 --gpu 0,1 --port 9100 --model orpheus --config configs/orpheus.yaml --concs 1,16,32,64 --save-audio "" --out $R > $R.base2g_r2.log 2>&1
echo 2G DONE
