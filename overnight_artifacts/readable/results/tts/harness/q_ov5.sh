cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_cbt.out; do sleep 10; done
for r in 1 2; do
$P bench.py --tree /root/work/exp --label ov5_graph_r$r --model omnivoice --gpus 0 --port 8100 --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/exp5 --label ov5_fix_r$r --model omnivoice --gpus 0 --port 8100 --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/exp5 --label ov5_fixsplit_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps --gpus 0 --port 8100 --points 1:ov25:1,8:ov150:1,32:ov150:2
done
$P bench.py --tree /root/work/exp5 --label ov5_fixsplit_phase --model omnivoice --config configs/omnivoice_split.yaml --mps --gpus 0 --port 8100 --phase 100 --points 32:ov150:1 --spy c32_ov150 --spy-delay 15 --spy-seconds 10 --no-audio
echo QUEUE DONE
