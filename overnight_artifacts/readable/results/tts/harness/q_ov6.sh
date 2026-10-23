cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_q2.out; do sleep 10; done
for r in 1 2; do
$P bench.py --tree /root/work/exp7 --label ov6_b32_r$r --model omnivoice --config /root/work/tharness/omnivoice_split_b32.yaml --mps --gpus 0 --port 8100 --points 8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/exp7 --label ov6_b16_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps --gpus 0 --port 8100 --points 8:ov150:1,32:ov150:2
done
echo QUEUE DONE
