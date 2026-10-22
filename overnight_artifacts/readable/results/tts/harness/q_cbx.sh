cd /root/guest-tts/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 9100"
PT="1:s26:1,8:s156:1,32:s156:2"
for m in turbo multilingual; do
$P bench.py --tree /root/guest-tts/base --label cbx_${m}_base_r1 --model chatterbox_$m --config configs/chatterbox_$m.yaml $B --points $PT
$P bench.py --tree /root/guest-tts/head --label cbx_${m}_head_r1 --model chatterbox_$m --config configs/chatterbox_$m.yaml $B --points $PT
$P bench.py --tree /root/guest-tts/head --label cbx_${m}_split_r1 --model chatterbox_$m --config configs/chatterbox_${m}_split.yaml --mps $B --points $PT
done
for m in turbo multilingual; do
$P bench.py --tree /root/guest-tts/base --label cbx_${m}_base_r2 --model chatterbox_$m --config configs/chatterbox_$m.yaml $B --points $PT
$P bench.py --tree /root/guest-tts/head --label cbx_${m}_split_r2 --model chatterbox_$m --config configs/chatterbox_${m}_split.yaml --mps $B --points $PT
done
echo CB DONE
for r in 1 2; do
$P bench.py --tree /root/guest-tts/ovloop --label ovx_loop_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 1:ov25:1,8:ov150:1,32:ov150:1
$P bench.py --tree /root/guest-tts/head --label ovx_head_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 1:ov25:1,8:ov150:1,32:ov150:1
done
$P bench.py --tree /root/guest-tts/head --label cbx_turbo_split_phase --model chatterbox_turbo --config configs/chatterbox_turbo_split.yaml --mps $B --phase 100 --no-audio --points 32:s156:1
$P bench.py --tree /root/guest-tts/ovloop --label ovx_loop_phase --model omnivoice --config configs/omnivoice_split.yaml --mps $B --phase 100 --no-audio --points 32:ov150:1
echo QUEUE DONE
