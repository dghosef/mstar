#!/bin/bash
# Idempotent box setup for the diffusion lane (portable: run on any box with the trees at /root/work/{mstar,mstar-main}).
# mstar venvs (setup_box.sh), pinned competitor venvs, the client env, prompt sets, diffusers hub kernels.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/root/work/hf
HERE=$(cd "$(dirname "$0")" && pwd)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
for t in mstar mstar-main; do
  [ -f /root/work/$t/.setup_done ] || { bash $HERE/setup_box.sh /root/work/$t > /root/work/setup_$t.log 2>&1 && touch /root/work/$t/.setup_done; }
done
E=/root/work/engines; mkdir -p $E; cd $E
( [ -x vllmomni/bin/python ] && vllmomni/bin/python -c "import vllm_omni" 2>/dev/null ) || {
  uv venv --python 3.12 --managed-python --seed vllmomni &&
  uv pip install --python vllmomni/bin/python --torch-backend=auto "vllm==0.30.0" &&
  uv pip install --python vllmomni/bin/python --torch-backend=auto "vllm-omni==0.30.0" ninja aiohttp; } > $E/vllmomni_install.log 2>&1 &
( [ -x sgld/bin/python ] && sgld/bin/python -c "import sglang.multimodal_gen" 2>/dev/null ) || {
  uv venv --python 3.12 --managed-python --seed sgld &&
  uv pip install --python sgld/bin/python --torch-backend=auto "sglang[diffusion]==0.5.22" ninja aiohttp; } > $E/sgld_install.log 2>&1 &
( [ -x diffusers/bin/python ] && diffusers/bin/python -c "import diffusers, av, skimage" 2>/dev/null ) || {
  uv venv --python 3.12 --managed-python --seed diffusers &&
  uv pip install --python diffusers/bin/python --torch-backend=cu132 "torch==2.14.1" "torchvision" &&
  uv pip install --python diffusers/bin/python "diffusers==0.41.0" "transformers" accelerate "kernels==0.17.2" sentencepiece ftfy imageio imageio-ffmpeg protobuf scikit-image av; } > $E/diffusers_install.log 2>&1 &
( [ -x /root/work/clientenv/bin/python ] && /root/work/clientenv/bin/python -c "import av, numpy, PIL" 2>/dev/null ) || {
  uv venv --python 3.12 --managed-python --seed /root/work/clientenv &&
  uv pip install --python /root/work/clientenv/bin/python numpy pillow av; } > $E/clientenv_install.log 2>&1 &
wait
mkdir -p /root/work/data; cd /root/work/data
[ -s parti64.txt ] || /root/work/mstar/.venv/bin/python - <<'PY'
from datasets import load_dataset
ds = load_dataset("nateraw/parti-prompts", split="train")
p = [ds[i]["Prompt"] for i in range(0, len(ds), 25)][:64]
open("parti64.txt", "w").write("\n".join(x.replace("\n", " ").strip() for x in p) + "\n")
PY
[ -s vbench_all_dimension.txt ] || curl -sfL https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/all_dimension.txt -o vbench_all_dimension.txt
[ -s vbench16.txt ] || awk 'NR%59==1' vbench_all_dimension.txt | head -16 > vbench16.txt
HF_TOKEN=$(cat /root/work/hf/token 2>/dev/null) /root/work/engines/diffusers/bin/python -c "
from kernels import get_kernel; get_kernel('kernels-community/flash-attn3', version=1)" > /dev/null 2>&1 || echo "fa3 hub kernel fetch failed"
/root/work/mstar/.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('openai/clip-vit-large-patch14')" >/dev/null 2>&1
echo "INSTALL_ENGINES DONE $(date -u)"
