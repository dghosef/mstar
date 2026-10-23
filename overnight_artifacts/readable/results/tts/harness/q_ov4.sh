cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_ov3.out; do sleep 10; done
$P bench.py --tree /root/work/exp --label ov4_gall_spy --model omnivoice --config /root/work/cfg/omnivoice_allbuckets.yaml --gpus 0 --port 8100 --points 32:ov150:1 --spy c32_ov150 --spy-delay 15 --spy-seconds 10 --no-audio
echo QUEUE DONE
