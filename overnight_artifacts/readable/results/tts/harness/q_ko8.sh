cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
B="--gpus 0 --port 8100"
until grep -q "QUEUE DONE" /root/work/q_cb6.out; do sleep 10; done
KP="--points 1:s26:1,32:s156:2,64:s1248:1"
for r in 1 2; do
$P bench.py --tree /root/work/head3 --label ko8_bf16_r$r --model kokoro --config /root/work/tharness/kokoro_bf16.yaml $B $KP
$P bench.py --tree /root/work/head3 --label ko8_head_r$r --model kokoro $B $KP
done
R=/workspace/results
CUDA_VISIBLE_DEVICES=1 HF_HOME=/root/work/hf $P quality.py --device cuda:0 --sentences s156.txt --base $R/ko8_head_r1/c32_s156_wav --test $R/ko8_bf16_r1/c32_s156_wav --out $R/quality/ko_bf16_vs_head.json > $R/quality/ko_bf16_vs_head.txt 2>&1
echo QUEUE DONE
