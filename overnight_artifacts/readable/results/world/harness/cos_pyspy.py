"""py-spy the API server process during a Cosmos3 DROID action session."""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from srv import Server  # noqa: E402

arm, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
srv = Server(arm, ["cosmos3_edge_droid", "--config", "configs/cosmos3_edge_droid.yaml"], 9142, str(out / "server.log"),
             gpus="0", timeout=3600)
srv.start()
try:
    # warm session
    subprocess.run([sys.executable, str(Path(__file__).parent / "ab_cosmos_session.py"), "9142", "6"], check=True)
    procs = subprocess.run(["ps", "-o", "pid=,cmd=", "--ppid", str(srv.proc.pid)], capture_output=True, text=True).stdout
    (out / "children.txt").write_text(procs)
    rec = subprocess.Popen(["/root/work/mstar/.venv/bin/py-spy", "record", "--pid", str(srv.proc.pid), "--rate", "50", "--nonblocking",
                            "--duration", "25", "--format", "raw", "--output", str(out / "api.txt"), "--threads"])
    time.sleep(1)
    subprocess.run([sys.executable, str(Path(__file__).parent / "ab_cosmos_session.py"), "9142", "24"], check=True)
    rec.wait()
finally:
    srv.stop()
