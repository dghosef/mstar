cd /root/work/bench
PY=/root/work/clientenv/bin/python
until grep -q CHAIN_A_DONE /workspace/results/bench/q35_0.8b_chainA.log; do sleep 15; done
$PY bench.py main --model q35_0.8b --arms mstar-pr --rounds 1,2,3
$PY bench.py xcheck --model q35_0.8b --arms mstar-pr,mstar-main,vllm,sglang
$PY bench.py profile --model q35_0.8b --points random_1024_256:128,random_1024_256:32,sharegpt:128,sharegpt:1
$PY bench.py summarize --model q35_0.8b
echo CHAIN_B_DONE
