cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
until ! pgrep -f "label hg_dbg2" > /dev/null; do sleep 10; done
for r in 1 2; do
$P bench.py --tree /root/work/ovB --label ov11_canvas0_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 1:ov25:1,8:ov150:1,32:ov150:2
$P bench.py --tree /root/work/ovA --label ov11_prev_r$r --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 1:ov25:1,8:ov150:1,32:ov150:2
done
for t in ovB ovA; do $P bench.py --tree /root/work/$t --label ov11_${t}_seed --model omnivoice --config configs/omnivoice_split.yaml --mps $B --points 32:ov150:1 --extra seed=1234; done
R=/workspace/results
CUDA_VISIBLE_DEVICES=1 HF_HOME=/root/work/hf $P quality.py --device cuda:0 --sentences ov150.txt --base $R/ov11_ovA_seed/c32_ov150_wav --test $R/ov11_ovB_seed/c32_ov150_wav --out $R/quality/ov_initcanvas_seed.json > $R/quality/ov_initcanvas_seed.txt 2>&1
echo QUEUE DONE
