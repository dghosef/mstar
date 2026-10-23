#!/bin/bash
# stress_run.sh <label> <model> <config> <port> <WxH> <frames>   (env passed through; run under the batch lock)
L=$1; M=$2; C=$3; P=$4; S=$5; F=$6
O=/workspace/results/video/$L; mkdir -p $O
cd /root/work/mstar && source .venv/bin/activate && export HF_HOME=/root/work/hf
setsid python -m mstar.cli.main serve $M --config $C --gpus 0,1 --port $P > $O/server.log 2>&1 < /dev/null &
SP=$!
timeout 3600 bash -c "until curl -sf http://127.0.0.1:$P/health >/dev/null; do sleep 3; done" || echo "never healthy"
timeout 1500 python /root/work/vharness/lockstep_stress.py $P $S $F 3 > $O/stress.txt 2>&1; echo "stress rc=$?" >> $O/stress.txt
kill -INT -- -$SP; sleep 20; kill -9 -- -$SP 2>/dev/null; sleep 5
echo "tracebacks: $(grep -c Traceback $O/server.log)" >> $O/stress.txt
