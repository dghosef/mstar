cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_g1.out; do sleep 10; done
$P bench.py --asr --tree /root/work/exp3 --label hg_merge_phase --model higgs_audio --gpus 1 --port 8200 --phase 100 --points 32:200:_,1:50:_,32:200:_
$P bench.py --asr --tree /root/work/exp --label hg_new_phase --model higgs_audio --gpus 1 --port 8200 --phase 100 --points 32:200:_,1:50:_,32:200:_
echo QUEUE DONE
