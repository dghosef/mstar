#!/bin/bash
# One model end to end, resumable (bench.py skips finished stages via markers).
#   [WAIT_LOG=<log> WAIT_TEXT=<text>] bash chain_model.sh <model> [tune knobs]
# Env: MTP_ROUNDS (default 1,2), PROFILE_POINTS, FRONTEND (python|rust: skip the frontend tune).
M=$1; KNOBS=${2:-default,lm-only,mnbt-16k,attn-fa3,chunk-16k}
cd /root/work/bench
PY=/root/work/clientenv/bin/python
D=/workspace/results/bench/$M
if [ -n "${WAIT_LOG:-}" ]; then until grep -q "${WAIT_TEXT}" "$WAIT_LOG" 2>/dev/null; do sleep 20; done; fi
echo "CHAIN $M start $(date -u)"
/root/work/engines/vllm/bin/python bench.py data --model $M
$PY bench.py correct --model $M --arms vllm,sglang,mstar-main,mstar-pr
$PY bench.py tune --model $M --arms vllm,sglang --knobs $KNOBS
if [ -z "${FRONTEND:-}" ]; then
  $PY bench.py fetune --model $M --arms mstar-pr
  $PY bench.py oracle --model $M
  read MAIN PR < <($PY decide_frontend.py $D)
else
  $PY bench.py oracle --model $M
  $PY bench.py setfrontend --model $M --knobs $FRONTEND
  if [ "$FRONTEND" = rust ]; then MAIN=mstar-main-rf; PR=mstar-pr-rf; else MAIN=mstar-main; PR=mstar-pr; fi
fi
echo "FRONTEND choice: $MAIN $PR"
grep -q "GATE FAIL" $D/gate.txt && echo "WARNING: a gate failed, see gate.txt"
$PY bench.py main --model $M --arms $MAIN,vllm,sglang --rounds 1,2,3
$PY bench.py mtp --model $M --arms vllm,sglang --rounds ${MTP_ROUNDS:-1,2} --concs 1,4,16
$PY bench.py main --model $M --arms $PR --rounds 1,2,3
$PY bench.py xcheck --model $M --arms $PR,$MAIN,vllm,sglang
$PY bench.py profile --model $M --arm $PR --points ${PROFILE_POINTS:-random_1024_256:128,random_1024_256:32,sharegpt:128,sharegpt:1}
$PY bench.py summarize --model $M
echo "CHAIN $M DONE $(date -u)"
