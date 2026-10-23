#!/bin/bash
# g_orph_pair.sh <label-suffix> <treeA:labelA:gpuA:portA> [<treeB:labelB:gpuB:portB>]
cd /root/guest-artts/harness
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf GUEST_PYLIB=/root/guest-artts/pylib GUEST_INDUCTOR=/root/guest-artts/inductor
R=/workspace/results/artts/orpheus
mkdir -p $R
shift 0
for spec in "$@"; do
  IFS=: read T L G P <<< "$spec"
  python bench_arm.py --tree /root/guest-artts/$T --label $L --gpu $G --port $P --model orpheus --config configs/orpheus_colocated.yaml --concs 1,16,32,64 --save-audio 32 --greedy temperature=0 --out $R > $R.$L.log 2>&1 &
done
wait
echo PAIR DONE
