#!/bin/bash
# nsys_run.sh <wait-script> <tree> <gpu> <port> <out-name> <serve-args...> -- <client args...>
# Profiles only the measured client run, after the server's warmup.
WAIT=$1; TREE=$2; GPU=$3; PORT=$4; NAME=$5; shift 5
SERVE=(); while [ "$1" != "--" ]; do SERVE+=("$1"); shift; done; shift
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf PYTHONPATH=$TREE MSTAR_SHM_ARENA=1 CUDA_HOME=/usr/local/cuda-12.9 TORCHINDUCTOR_CACHE_DIR=/root/work/inductor_cache TORCHINDUCTOR_COMPILE_THREADS=8
while pgrep -f "^/bin/bash /root/work/harness_artts/$WAIT" >/dev/null; do sleep 15; done
R=/workspace/results/nsys; mkdir -p $R
cd $TREE
setsid nsys launch --session-new=$NAME --trace=cuda,nvtx --cuda-graph-trace=node \
  python -m mstar.cli.main serve "${SERVE[@]}" --gpus $GPU --port $PORT > $R/${NAME}_server.log 2>&1 &
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do sleep 5; done
cd /root/work/harness_artts
python -m benchmark.tts_speech_bench --engine mstar --url http://127.0.0.1:$PORT "$@" --repeats 1 --warmup 64 --num-sentences 16 > $R/${NAME}_warm.log 2>&1
nsys start --session=$NAME -o $R/$NAME -f true
python -m benchmark.tts_speech_bench --engine mstar --url http://127.0.0.1:$PORT "$@" --repeats 1 --warmup 0 > $R/${NAME}_client.log 2>&1
nsys stop --session=$NAME
sleep 5
pkill -INT -f "serve .* --port $PORT\$" ; sleep 60
nsys stats --report cuda_gpu_kern_sum --format csv -o $R/${NAME} $R/$NAME.nsys-rep > /dev/null 2>&1
echo NSYS DONE $NAME
