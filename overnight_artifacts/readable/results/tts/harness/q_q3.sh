cd /root/work/tharness
P=/root/work/mstar/.venv/bin/python
export HF_HOME=/root/work/hf CUDA_VISIBLE_DEVICES=1
until grep -q "QUEUE DONE" /root/work/q_ko2.out; do sleep 10; done
R=/workspace/results
$P quality.py --device cuda:0 --sentences s156.txt --base $R/ko_base_r2/c32_s156_wav --test $R/ko_bf16_r2/c32_s156_wav --out $R/quality/ko_bf16_vs_fp32.json > $R/quality/ko_bf16_vs_fp32.txt 2>&1
$P quality.py --device cuda:0 --sentences s156.txt --base $R/ko_base_r1/c32_s156_wav --test $R/ko_base_r2/c32_s156_wav --out $R/quality/ko_fp32_vs_fp32.json > $R/quality/ko_fp32_vs_fp32.txt 2>&1
echo QUEUE DONE
