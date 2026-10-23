#!/bin/bash
# Box-side provisioning after the trees are rsynced to /root/work/{mstar,mstar-main} and the
# harness to /root/work/bench (Mac side: harness/push.sh <host>).
#   bash /root/work/bench/provision.sh   (log: /root/work/provision.log)
# Every install / download is idempotent, so a rerun only fills what's missing.
set -uo pipefail
export HF_HOME=/root/work/hf
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
mkdir -p /root/work/hf /root/work/engines /workspace/results/bench
cd /root/work
# 1. both mstar trees: venv + editable install + Rust extension (setup_box.sh)
for t in ${TREES_TO_SETUP:-mstar mstar-main}; do
  [ -f /root/work/$t/.setup_done ] && continue
  bash /root/work/bench/setup_box.sh /root/work/$t && touch /root/work/$t/.setup_done
done
# 2. pinned competitors
cd /root/work/engines
[ -x vllm/bin/vllm ] || { uv venv --python 3.12 --managed-python --seed vllm &&
  uv pip install --python vllm/bin/python --torch-backend=auto "vllm==0.32.0" ninja aiohttp datasets pandas; }
[ -x sglang/bin/python ] && sglang/bin/python -c "import sglang" 2>/dev/null || { uv venv --python 3.12 --managed-python --seed sglang &&
  uv pip install --python sglang/bin/python --torch-backend=auto "sglang==0.5.22" ninja aiohttp datasets pandas; }
# vLLM asks for its precompiled FlashInfer kernels (else it JIT-compiles them at startup)
[ -f vllm/.kernels_done ] || { vllm/bin/vllm download-kernels && touch vllm/.kernels_done; } || echo "download-kernels failed"
# shared client env, separate from every arm's venv
[ -x /root/work/clientenv/bin/python ] || { uv venv --python 3.12 --managed-python --seed /root/work/clientenv &&
  uv pip install --python /root/work/clientenv/bin/python aiohttp uvloop orjson "transformers==5.20.2" datasets numpy jinja2; }
# 3. weights, datasets, assets (servers run with HF_HUB_OFFLINE=1)
PY=/root/work/mstar/.venv/bin/python
for m in ${MODELS:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B}; do
  $PY -c "from huggingface_hub import snapshot_download as s; s('$m')" && echo "DL_OK $m"
done
$PY -c "
from huggingface_hub import hf_hub_download as h
h('anon8231489123/ShareGPT_Vicuna_unfiltered','ShareGPT_V3_unfiltered_cleaned_split.json',repo_type='dataset')
for f in ['assets/example_action_fd_umi_first_frame.png','assets/edge_action_id_av_0_output.png']:
    h('nvidia/Cosmos3-Edge', f)
from datasets import load_dataset; load_dataset('openai/gsm8k','main')
print('DL_OK data')"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv > /workspace/results/bench/hardware.txt
nvcc --version | tail -1 >> /workspace/results/bench/hardware.txt
lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)" >> /workspace/results/bench/hardware.txt
echo "PROVISION DONE $(date)"
