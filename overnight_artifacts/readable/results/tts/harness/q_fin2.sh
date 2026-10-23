cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
until grep -q "QUEUE DONE" /root/work/q_clone2.out; do sleep 10; done
for r in 1 2; do
$P bench.py --tree /root/work/base --label f2_ko_base_r$r --model kokoro $B --points 1:s26:1,32:s156:2,64:s1248:1
$P bench.py --tree /root/work/head4 --label f2_ko_head_r$r --model kokoro $B --points 1:s26:1,32:s156:2,64:s1248:1
$P bench.py --asr --tree /root/work/base --label f2_hg_base_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_
$P bench.py --asr --tree /root/work/head4 --label f2_hg_head_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_
$P bench.py --tree /root/work/base --label f2_ov_base_r$r --model omnivoice $B --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/head4 --label f2_ov_head_r$r --model omnivoice $B --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/head4 --label f2_ov_split_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/base --label f2_cb_base_r$r --model chatterbox $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/head4 --label f2_cb_head_r$r --model chatterbox $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/head4 --label f2_cb_split_r$r --model chatterbox --config configs/chatterbox_split.yaml --mps $B --points 1:s26:1,8:s156:1,32:s156:2
done
echo QUEUE DONE
