import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from srv import Server
arm, out, port, gpu = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
srv = Server(arm, ["cosmos3_edge_droid", "--config", "configs/cosmos3_edge_droid.yaml"], int(port), str(out / "server.log"), gpus=gpu, timeout=3600)
srv.start()
try:
    subprocess.run([sys.executable, str(Path(__file__).parent / "ab_cosmos_session.py"), port, "16"], check=True)
finally:
    srv.stop()
