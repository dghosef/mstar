#!/bin/bash
# Integrator box prep, after setup_box.sh: test deps, Kokoro tagger, weights, arm dirs.
set -uo pipefail
export HF_HOME=/root/work/hf PATH="/root/work/mstar/.venv/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
until grep -q "SETUP DONE" /root/work/setup.log 2>/dev/null; do
  if grep -q -E "^(Traceback|error:)" /root/work/setup.log 2>/dev/null && ! pgrep -f setup_box.sh >/dev/null; then echo "SETUP FAILED"; exit 1; fi
  sleep 20
done
PY=/root/work/mstar/.venv/bin/python
uv pip install --python $PY "pytest-xdist==3.8.0" "audioseal>=0.2,<0.3" jiwer soundfile aiohttp pillow
$PY -m spacy download en_core_web_sm || uv pip install --python $PY "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
$PY - <<'PY'
from huggingface_hub import snapshot_download
import concurrent.futures as cf
jobs = [
    ("Qwen/Qwen3.5-0.8B", None),
    ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", None),
    ("Tongyi-MAI/Z-Image-Turbo", None),
    ("hexgrad/Kokoro-82M", None),
    ("openai/whisper-large-v3-turbo", None),
    ("nvidia/Cosmos3-Edge", ["assets/example_action_fd_umi_first_frame.png", "assets/edge_action_id_av_0_output.png"]),
]
def dl(j):
    repo, pats = j
    try:
        p = snapshot_download(repo, allow_patterns=pats, max_workers=16)
        return f"OK {repo} {p}"
    except Exception as e:
        return f"FAIL {repo} {e!r}"[:300]
with cf.ThreadPoolExecutor(6) as ex:
    for r in ex.map(dl, jobs):
        print(r, flush=True)
PY
mkdir -p /root/work/arms /workspace/results/integrator
echo "PREP DONE $(date)"
