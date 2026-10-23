cd /root/work/bench
PY=/root/work/clientenv/bin/python
while pgrep -f "bench[.]py tune" >/dev/null; do sleep 10; done
$PY bench.py correct --model q35_0.8b --arms mstar-pr
$PY bench.py oracle --model q35_0.8b
$PY bench.py main --model q35_0.8b --arms mstar-main,vllm,sglang --rounds 1,2,3
$PY bench.py mtp --model q35_0.8b --arms vllm,sglang --rounds 1,2,3
echo CHAIN_A_DONE
