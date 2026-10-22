import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from srv import PY, Server  # noqa: E402

arm, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "vjepa2_ac"
extra = sys.argv[4:]
Path(out).mkdir(parents=True, exist_ok=True)
import os
env = {k: v for k, v in os.environ.items() if k.startswith("MSTAR_RUST") or k == "MSTAR_SHM_ARENA"}
srv = Server(arm, [model, "--config", f"configs/{model}.yaml"], 9116, f"{out}/server.log", gpus="1", env=env)
srv.start()
try:
    subprocess.run([PY, str(Path(__file__).parent / "ac_stream_vs_batch.py"), "9116", "/root/work/media/droid_like_8f.mp4" if model == "vjepa2_ac" else "/root/work/mstar/test/qwen3-omni/video.webm",
                    *(extra or [str(Path(__file__).parent / "ac_kwargs_h4.json")])], check=True)
finally:
    srv.stop()
