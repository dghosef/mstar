cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_g0b.out; do sleep 10; done
for r in 1 2; do
$P bench.py --tree /root/work/exp6 --label cb_vc_r$r --model chatterbox --config /root/work/cfg/chatterbox_split_vc.yaml --mps --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
$P bench.py --tree /root/work/exp6 --label cb_novc_r$r --model chatterbox --config /root/work/cfg/chatterbox_split.yaml --mps --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
done
echo QUEUE DONE
