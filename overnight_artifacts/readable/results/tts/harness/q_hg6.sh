cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
until grep -q "QUEUE DONE" /root/work/q_fin2.out; do sleep 10; done
for r in 1 2; do
$P bench.py --asr --tree /root/work/exp14 --label hg6_mixed_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_
$P bench.py --asr --tree /root/work/head4 --label hg6_prev_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_
done
$P bench.py --asr --tree /root/work/exp14 --label hg6_mixed_phase --model higgs_audio $B --phase 100 --points 32:200:_,32:200:_
echo QUEUE DONE
