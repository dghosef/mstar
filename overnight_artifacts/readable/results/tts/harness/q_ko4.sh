cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_ko3.out; do sleep 10; done
for r in 1 2; do
$P bench.py --tree /root/work/exp9 --label ko4_bench_r$r --model kokoro --gpus 1 --port 8200 --points 1:s26:1,32:s156:2,64:s1248:1
$P bench.py --tree /root/work/exp8 --label ko4_fuse_r$r --model kokoro --gpus 1 --port 8200 --points 1:s26:1,32:s156:2,64:s1248:1
done
echo QUEUE DONE
