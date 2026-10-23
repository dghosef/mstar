cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100 --mps"
until ! pgrep -f "label hg5_phase" >/dev/null; do sleep 10; done
for r in 1 2; do
$P bench.py --tree /root/work/head3 --label cb6_s4_r$r --model chatterbox --config /root/work/cfg/chatterbox_split_s4.yaml $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/head3 --label cb6_s8_r$r --model chatterbox --config configs/chatterbox_split.yaml $B --points 1:s26:1,8:s156:1,32:s156:2
done
echo QUEUE DONE
