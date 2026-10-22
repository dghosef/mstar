cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
until grep -q "QUEUE DONE" /root/work/q_cb7.out 2>/dev/null; do sleep 10; done
$P bench.py --tree /root/work/exp16 --label cb7_rows_seed --model chatterbox $B --extra seed=0 --points 1:s26:1,8:s156:1
$P bench.py --tree /root/work/head4 --label cb7_prev_seed --model chatterbox $B --extra seed=0 --points 1:s26:1,8:s156:1
echo QUEUE DONE
