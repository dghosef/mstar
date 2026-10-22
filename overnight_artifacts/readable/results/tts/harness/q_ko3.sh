cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
for r in 1 2; do
$P bench.py --tree /root/work/exp8 --label ko3_fuse_r$r --model kokoro --gpus 1 --port 8200 --points 1:s26:1,32:s156:2,64:s1248:1
$P bench.py --tree /root/work/base --label ko3_base_r$r --model kokoro --gpus 1 --port 8200 --points 1:s26:1,32:s156:2,64:s1248:1
done
$P bench.py --tree /root/work/exp8 --label ko3_fuse_phase --model kokoro --gpus 1 --port 8200 --phase 100 --points 64:s1248:1 --no-audio
echo QUEUE DONE
