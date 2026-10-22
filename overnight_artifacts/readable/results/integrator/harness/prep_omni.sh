#!/bin/bash
# Qwen3-Omni prerequisites on the integrator box: CUDA 13 toolkit for the flash-attn build, flash-attn, weights.
set -uo pipefail
export HF_HOME=/root/work/hf
PY=/root/work/mstar/.venv/bin/python
(
  $PY -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-Omni-30B-A3B-Instruct', max_workers=16))"
  echo "OMNI WEIGHTS DONE"
) > /root/work/omni_weights.log 2>&1 &
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-0 > /root/work/cuda13.log 2>&1 || echo "apt cuda13 failed"
update-alternatives --set cuda /usr/local/cuda-12.9 2>/dev/null || true
CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:/root/work/mstar/.venv/bin:$PATH FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=32 \
  nice -n 15 uv pip install --python $PY flash-attn==2.8.3.post1 --no-build-isolation && echo "FLASH ATTN DONE"
wait
$PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
echo "OMNI PREP DONE"
