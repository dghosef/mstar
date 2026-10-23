#!/bin/bash
# Generic mstar setup on an Eras H200 slice (root in a container).
# Tree + venv + HF cache on the local NVMe (/root/work, fast, lost on terminate);
# results on the persistent volume (/workspace/results).
#   rsync the tree to <box>:/root/work/mstar, then: ssh <box> 'bash /root/work/harness/setup_box.sh'
set -euo pipefail
WORK=${1:-/root/work/mstar}   # tree dir; guests on a shared box pass their own, e.g. /root/guest-video/mstar
mkdir -p /root/work/hf /workspace/results
export HF_HOME=/root/work/hf
grep -q "HF_HOME=/root/work/hf" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'RC'
export HF_HOME=/root/work/hf
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
RC
cd "$WORK"
export PATH="$HOME/.local/bin:$PATH"
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libzmq3-dev pkg-config ffmpeg libsndfile1 sox iproute2 build-essential wget rsync git-lfs numactl >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x .venv/bin/python ] || uv venv --python 3.12 --managed-python --seed .venv
uv pip install --python .venv/bin/python --torch-backend=auto -e ".[all]" pytest ninja aiohttp jiwer soundfile py-spy
uv pip install --python .venv/bin/python --torch-backend=cu132 --reinstall-package torch --reinstall-package torchvision "torch==2.12.1" "torchvision==0.27.1"
uv pip install --python .venv/bin/python --torch-backend=cpu --reinstall-package torchaudio --no-deps "torchaudio==2.11.0"
uv pip install --python .venv/bin/python --no-deps "qwen-tts==0.1.1" || true
command -v cargo >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "${CARGO_HOME:-$HOME/.cargo}/env"
uv pip install --python .venv/bin/python maturin
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --release -m rust/Cargo.toml
.venv/bin/python -c "import torch, mstar_rust, mstar; assert torch.cuda.is_available(); print('env OK', torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
# CUDA toolkit: never let a 13.x apt package repoint /usr/local/cuda (breaks FlashInfer JIT)
if [ -d /usr/local/cuda-12.9 ]; then update-alternatives --set cuda /usr/local/cuda-12.9 2>/dev/null || true; fi
echo "SETUP DONE $(date)"
