"""Start a vjepa2_ac server for an arm and run ac_batch_probe against it."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from srv import PY, Server  # noqa: E402

arm, out, gpus = sys.argv[1], sys.argv[2], sys.argv[3]
env = dict(a.split("=", 1) for a in sys.argv[4:])
Path(out).mkdir(parents=True, exist_ok=True)
srv = Server(arm, ["vjepa2_ac", "--config", "configs/vjepa2_ac.yaml"], 9115, f"{out}/server.log", gpus=gpus, env=env)
srv.start()
try:
    subprocess.run([PY, str(Path(__file__).parent / "ac_batch_probe.py"), "9115", "/root/work/media/droid_like_8f.mp4",
                    str(Path(__file__).parent / "ac_kwargs_h4.json"), out, "3"], check=True)
finally:
    srv.stop()
