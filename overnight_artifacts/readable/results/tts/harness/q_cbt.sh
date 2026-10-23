cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
until grep -q "QUEUE DONE" /root/work/q_ov4.out; do sleep 10; done
$P bench.py --tree /root/work/mstar --label cbt_base_r1 --model chatterbox_turbo --gpus 0 --port 8100 --phase 100 --points 1:s26:1,32:s156:2
$P bench.py --tree /root/work/mstar --label cbt_splitmps_r1 --model chatterbox_turbo --config /root/work/cfg/chatterbox_turbo_split.yaml --mps --gpus 0 --port 8100 --phase 100 --points 1:s26:1,32:s156:2
$P bench.py --tree /root/work/mstar --label cb_base_r2 --model chatterbox --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
$P bench.py --tree /root/work/mstar --label cb_splitmps_r2 --model chatterbox --config /root/work/cfg/chatterbox_split.yaml --mps --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
echo QUEUE DONE
