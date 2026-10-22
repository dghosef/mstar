cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_cb4.out; do sleep 10; done
for r in 1 2; do
$P bench.py --tree /root/work/exp5 --label ov8_fine_r$r --model omnivoice --config /root/work/tharness/omnivoice_split_fine.yaml --mps --gpus 0 --port 8100 --points 8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/exp5 --label ov8_cap_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps --gpus 0 --port 8100 --points 8:ov150:1,32:ov150:2
done
echo QUEUE DONE
