cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
while kill -0 633014 2>/dev/null; do sleep 5; done
for r in 1 2; do
$P bench.py --tree /root/work/exp16 --label cb8_rows_r$r --model chatterbox $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/head4 --label cb8_prev_r$r --model chatterbox $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/exp16 --label cb8_rowssplit_r$r --model chatterbox --config configs/chatterbox_split.yaml --mps $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/head4 --label cb8_prevsplit_r$r --model chatterbox --config configs/chatterbox_split.yaml --mps $B --points 1:s26:1,8:s156:1,32:s156:2
done
$P bench.py --tree /root/work/exp16 --label cb8_rows_phase --model chatterbox $B --phase 100 --points 32:s156:1
$P bench.py --tree /root/work/exp16 --label cb8_rows_seed --model chatterbox $B --extra seed=0 --points 1:s26:1,8:s156:1
$P bench.py --tree /root/work/head4 --label cb8_prev_seed --model chatterbox $B --extra seed=0 --points 1:s26:1,8:s156:1
echo QUEUE DONE
