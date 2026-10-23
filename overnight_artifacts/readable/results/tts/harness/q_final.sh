cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
for r in 1 2; do
$P bench.py --tree /root/work/base --label fin_cb_base_r$r --model chatterbox $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/head --label fin_cb_head_r$r --model chatterbox $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/head --label fin_cb_headsplit_r$r --model chatterbox --config configs/chatterbox_split.yaml --mps $B --points 1:s26:1,8:s156:1,32:s156:2
$P bench.py --tree /root/work/base --label fin_ov_base_r$r --model omnivoice $B --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/head --label fin_ov_head_r$r --model omnivoice $B --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/head --label fin_ov_headsplit_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --asr --tree /root/work/base --label fin_hg_base_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_
$P bench.py --asr --tree /root/work/head --label fin_hg_head_r$r --model higgs_audio $B --points 32:200:_,1:50:_,8:200:_,32:200:_
done
echo QUEUE DONE
