cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
until grep -q "QUEUE DONE" /root/work/q_ov9.out; do sleep 10; done
KP="--points 1:s26:1,32:s156:2,64:s1248:1"
for r in 1 2; do
$P bench.py --tree /root/work/head2 --label ko7_fuse_r$r --model kokoro $B $KP
$P bench.py --tree /root/work/exp10 --label ko7_bench_r$r --model kokoro $B $KP
done
$P bench.py --tree /root/work/head2 --label ko7_fusebf16_r1 --model kokoro --config /root/work/tharness/kokoro_bf16.yaml $B $KP
echo QUEUE DONE
