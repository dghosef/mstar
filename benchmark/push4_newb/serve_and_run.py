"""Start one arm's server (ab.Server: pinned, own port and caches), run a command against it, stop it.

  serve_and_run.py --arm name=/tree[:K=V;...] --model qwen3_5_0.8b --gpus 0 --port 18100 --out DIR -- <cmd ...>
The command may use {port}; its output goes to DIR/<name>_cmd.log, the server's to DIR/<name>_server.log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="qwen3_5_0.8b")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--port", type=int, default=59100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv[:cut])
    cmd = [x.replace("{port}", str(a.port)) for x in argv[cut + 1:]]
    name, tree, env = ab.parse_arms(a.arm)[0]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    s = ab.Server(name, tree, env, a.model, a.gpus, a.port, out / f"{name}_server.log")
    try:
        r = subprocess.run(ab.pin(cmd), capture_output=True, text=True)
        (out / f"{name}_cmd.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        print(r.stdout[-3000:])
    finally:
        s.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
