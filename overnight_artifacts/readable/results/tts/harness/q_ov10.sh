cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_hg3.out; do sleep 10; done
$P bench.py --tree /root/work/head2 --label ov10_spy --model omnivoice --config configs/omnivoice_split.yaml --mps --gpus 0 --port 8100 --phase 100 --points 32:ov150:1 --spy c32_ov150 --spy-delay 15 --spy-seconds 10 --no-audio
echo QUEUE DONE
