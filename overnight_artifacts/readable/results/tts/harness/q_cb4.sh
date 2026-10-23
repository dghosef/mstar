cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
export HF_HOME=/root/work/hf
until grep -q "QUEUE DONE" /root/work/q_ov7.out; do sleep 10; done
cp /root/work/mstar/configs/chatterbox_multilingual_split.yaml /root/work/cfg/
$P bench.py --tree /root/work/mstar --label cbm_base_r1 --model chatterbox_multilingual --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
$P bench.py --tree /root/work/mstar --label cbm_splitmps_r1 --model chatterbox_multilingual --config /root/work/cfg/chatterbox_multilingual_split.yaml --mps --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
$P bench.py --tree /root/work/mstar --label cbt_splitmps_r2 --model chatterbox_turbo --config /root/work/cfg/chatterbox_turbo_split.yaml --mps --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
$P bench.py --tree /root/work/mstar --label cbt_base_r2 --model chatterbox_turbo --gpus 0 --port 8100 --points 1:s26:1,32:s156:2
R=/workspace/results
export CUDA_VISIBLE_DEVICES=0
$P quality.py --device cuda:0 --sentences s156.txt --base $R/cbm_base_r1/c32_s156_wav --test $R/cbm_splitmps_r1/c32_s156_wav --out $R/quality/cbm_split_vs_base.json > $R/quality/cbm_split_vs_base.txt 2>&1
$P quality.py --device cuda:0 --sentences s156.txt --base $R/cbt_base_r2/c32_s156_wav --test $R/cbt_splitmps_r2/c32_s156_wav --out $R/quality/cbt_split_vs_base.json > $R/quality/cbt_split_vs_base.txt 2>&1
echo QUEUE DONE
