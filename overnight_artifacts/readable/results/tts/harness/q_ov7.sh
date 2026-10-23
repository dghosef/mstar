cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
export HF_HOME=/root/work/hf
until grep -q "QUEUE DONE" /root/work/q_ov6.out; do sleep 10; done
$P bench.py --tree /root/work/exp5 --label ov7_fix_seed --model omnivoice --gpus 0 --port 8100 --points 32:ov150:1 --extra seed=1234
$P bench.py --tree /root/work/exp5 --label ov7_fixsplit_seed --model omnivoice --config configs/omnivoice_split.yaml --mps --gpus 0 --port 8100 --points 32:ov150:1 --extra seed=1234
R=/workspace/results
CUDA_VISIBLE_DEVICES=0 $P quality.py --device cuda:0 --sentences ov150.txt --base $R/ov7_fix_seed/c32_ov150_wav --test $R/ov7_fixsplit_seed/c32_ov150_wav --out $R/quality/ov_fixsplit_vs_fix_seed.json > $R/quality/ov_fixsplit_vs_fix_seed.txt 2>&1
echo QUEUE DONE
