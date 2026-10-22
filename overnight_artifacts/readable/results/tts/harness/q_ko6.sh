cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
until grep -q "QUEUE DONE" /root/work/q_serial.out; do sleep 10; done
KP="--points 1:s26:1,32:s156:2,64:s1248:1"
for r in 1 2; do
$P bench.py --tree /root/work/exp9 --label ko6_fine_r$r --model kokoro --config /root/work/tharness/kokoro_finebuckets.yaml $B $KP
$P bench.py --tree /root/work/exp9 --label ko6_bench_r$r --model kokoro $B $KP
done
echo QUEUE DONE
