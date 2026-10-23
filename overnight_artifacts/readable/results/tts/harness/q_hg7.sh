cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
for r in 1 2; do
$P bench.py --asr --tree /root/work/exp15 --label hg7_loop_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_,32:600:_
$P bench.py --asr --tree /root/work/head4 --label hg7_prev_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_,32:600:_
done
$P bench.py --asr --tree /root/work/exp15 --label hg7_loop_phase --model higgs_audio $B --phase 100 --points 32:200:_,32:600:_
echo QUEUE DONE
