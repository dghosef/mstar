cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
export TORCHINDUCTOR_COMPILE_THREADS=8 MAX_JOBS=16
until grep -q "QUEUE DONE" /root/work/q_ko6.out; do sleep 10; done
for r in 1 2; do
uptime >> /root/work/q_ov9.load
$P bench.py --tree /root/work/head2 --label ov9_canvas_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/head --label ov9_prev_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 1:ov25:1,8:ov150:1,32:ov150:2
uptime >> /root/work/q_ov9.load
done
echo QUEUE DONE
