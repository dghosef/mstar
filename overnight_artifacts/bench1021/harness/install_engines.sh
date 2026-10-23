#!/bin/bash
# Box-side: pinned competitor venvs for the diffusion lane.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/root/work/hf
E=/root/work/engines; mkdir -p $E; cd $E
( [ -x vllmomni/bin/python ] && vllmomni/bin/python -c "import vllm_omni" 2>/dev/null ) || {
  uv venv --python 3.12 --managed-python --seed vllmomni &&
  uv pip install --python vllmomni/bin/python --torch-backend=auto "vllm==0.30.0" &&
  uv pip install --python vllmomni/bin/python --torch-backend=auto "vllm-omni==0.30.0" ninja aiohttp; } > $E/vllmomni_install.log 2>&1 &
( [ -x sgld/bin/python ] && sgld/bin/python -c "import sglang.multimodal_gen" 2>/dev/null ) || {
  uv venv --python 3.12 --managed-python --seed sgld &&
  uv pip install --python sgld/bin/python --torch-backend=auto "sglang[diffusion]==0.5.22" ninja aiohttp; } > $E/sgld_install.log 2>&1 &
( [ -x diffusers/bin/python ] && diffusers/bin/python -c "import diffusers" 2>/dev/null ) || {
  uv venv --python 3.12 --managed-python --seed diffusers &&
  uv pip install --python diffusers/bin/python --torch-backend=cu132 "torch==2.14.1" "torchvision" &&
  uv pip install --python diffusers/bin/python "diffusers==0.41.0" "transformers" accelerate "kernels==0.17.2" sentencepiece ftfy imageio imageio-ffmpeg protobuf scikit-image; } > $E/diffusers_install.log 2>&1 &
wait
for v in vllmomni sgld diffusers; do echo "== $v"; $E/$v/bin/python -c "import torch;print('torch',torch.__version__)" 2>&1 | tail -1; done
echo INSTALL_DONE $(date)
