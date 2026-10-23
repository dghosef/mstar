cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
export HF_HOME=/root/work/hf
until grep -q "QUEUE DONE" /root/work/q_ov8.out; do sleep 10; done
$P bench.py --tree /root/work/mstar --label cb_base_seed --model chatterbox --gpus 0 --port 8100 --points 32:s156:1 --extra seed=1234
$P bench.py --tree /root/work/mstar --label cb_split_seed --model chatterbox --config /root/work/cfg/chatterbox_split.yaml --mps --gpus 0 --port 8100 --points 32:s156:1 --extra seed=1234
R=/workspace/results
CUDA_VISIBLE_DEVICES=0 $P quality.py --device cuda:0 --sentences s156.txt --base $R/cb_base_seed/c32_s156_wav --test $R/cb_split_seed/c32_s156_wav --out $R/quality/cb_split_vs_base_seed.json > $R/quality/cb_split_vs_base_seed.txt 2>&1
echo QUEUE DONE
