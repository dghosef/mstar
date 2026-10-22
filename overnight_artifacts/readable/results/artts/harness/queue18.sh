#!/bin/bash
cd /root/work/harness_artts
export PATH=/root/work/mstar/.venv/bin:$PATH HF_HOME=/root/work/hf
python ab.py --arms base=/root/work/base2,head=/root/work/t_head3 --rounds 2 --out /workspace/results/qvd_final -- --model qwen3_tts_voicedesign --config configs/qwen3tts_voicedesign.yaml --gpu 0 --port 8100 --language English --extra "instructions=A calm, warm female voice with a slight smile, speaking at a measured pace." --concs 1,16,32,64 --save-audio 16
S=/root/work/t_head3/.claude/skills/benchmarking/scripts/server.sh
bash $S mps-start 0
MPS="--env CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_root_gpu0/pipe --env CUDA_MPS_LOG_DIRECTORY=/tmp/mps_root_gpu0/log"
COMMON="--model qwen3_tts_1p7b --gpu 0 --port 8100 --voice vivian --language English --concs 1,32,64 --save-audio 32 --out /workspace/results/q17_split"
for r in 1 2; do
  if [ $((r % 2)) = 1 ]; then
    python bench_arm.py --tree /root/work/t_head3 --label single_r$r --config configs/qwen3tts_1p7b.yaml $COMMON
    python bench_arm.py --tree /root/work/t_head3 --label splitmps_r$r --config configs/qwen3tts_1p7b_split.yaml $COMMON $MPS
  else
    python bench_arm.py --tree /root/work/t_head3 --label splitmps_r$r --config configs/qwen3tts_1p7b_split.yaml $COMMON $MPS
    python bench_arm.py --tree /root/work/t_head3 --label single_r$r --config configs/qwen3tts_1p7b.yaml $COMMON
  fi
done
bash $S mps-stop 0
echo AB DONE
