import sys
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download

jobs = {
    "qwen35": ("Qwen/Qwen3.5-0.8B", None),
    "cosmos": ("nvidia/Cosmos3-Edge", ["assets/example_action_fd_umi_first_frame.png", "assets/edge_action_id_av_0_output.png"]),
    "zimage": ("Tongyi-MAI/Z-Image-Turbo", None),
    "kokoro": ("hexgrad/Kokoro-82M", None),
    "whisper": ("openai/whisper-large-v3-turbo", None),
    "tts": ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", None),
}


def dl(k):
    repo, pats = jobs[k]
    try:
        return f"OK {repo} {snapshot_download(repo, allow_patterns=pats, max_workers=16)}"
    except Exception as e:  # noqa: BLE001
        return f"FAIL {repo} {e!r}"[:300]


with ThreadPoolExecutor(6) as ex:
    for r in ex.map(dl, sys.argv[1:]):
        print(r, flush=True)
print("DL DONE")
