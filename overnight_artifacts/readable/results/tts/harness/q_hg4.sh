cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
until grep -q "QUEUE DONE" /root/work/q_ov11.out; do sleep 10; done
for r in 1 2; do
$P bench.py --asr --tree /root/work/exp13 --label hg4_mask_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_
$P bench.py --asr --tree /root/work/ovA --label hg4_prev_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_
done
echo QUEUE DONE
