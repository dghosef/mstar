cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
for r in 1 2; do
$P bench.py --tree /root/work/mstar --label ko_base_r$r --model kokoro --gpus 1 --port 8200 --points 1:s26:1,32:s156:2,64:s1248:1
$P bench.py --tree /root/work/mstar --label ko_bf16_r$r --model kokoro --config /root/work/tharness/kokoro_bf16.yaml --gpus 1 --port 8200 --points 1:s26:1,32:s156:2,64:s1248:1
$P bench.py --tree /root/work/mstar --label ko_single_r$r --model kokoro --config /root/work/tharness/kokoro_single.yaml --gpus 1 --port 8200 --points 1:s26:1,32:s156:2,64:s1248:1
done
echo QUEUE DONE
