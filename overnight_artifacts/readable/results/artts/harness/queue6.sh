#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
while pgrep -f "^/bin/bash /root/work/harness_artts/queue5.sh" >/dev/null; do sleep 15; done
rsync -a /root/work/t_q5/ /root/work/t_q5s/ && cp /root/work/t_q4/configs/qwen3tts_split.yaml /root/work/t_q5s/configs/
S=/root/work/t_q5s/.claude/skills/benchmarking/scripts/server.sh
bash $S mps-start 0
MPS="--env CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_root_gpu0/pipe --env CUDA_MPS_LOG_DIRECTORY=/tmp/mps_root_gpu0/log"
COMMON="--model qwen3_tts --gpu 0 --port 8100 --voice vivian --language English --concs 1,32,64 --save-audio 32 --out /workspace/results/q06_ab6"
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then
    python bench_arm.py --tree /root/work/t_q5 --label single_r$r --config configs/qwen3tts.yaml $COMMON
    python bench_arm.py --tree /root/work/t_q5s --label splitmps_r$r --config configs/qwen3tts_split.yaml $COMMON $MPS
  else
    python bench_arm.py --tree /root/work/t_q5s --label splitmps_r$r --config configs/qwen3tts_split.yaml $COMMON $MPS
    python bench_arm.py --tree /root/work/t_q5 --label single_r$r --config configs/qwen3tts.yaml $COMMON
  fi
done
bash $S mps-stop 0
echo AB DONE
