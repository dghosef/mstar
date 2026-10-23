cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_ov10.out; do sleep 10; done
$P bench.py --asr --tree /root/work/exp11 --label hg_dbg --model higgs_audio --gpus 0 --port 8100 --points 32:200:_,32:200:_
echo QUEUE DONE
