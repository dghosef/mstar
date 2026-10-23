"""A/B a vjepa2 / vjepa2_ac server across arms, alternating arms per round.

  python ab_vjepa2.py --model vjepa2 --arms base=/root/work/base new=/root/work/mstar \
      --rounds 2 --cells 1:20 4:40 --out /workspace/results/vjepa2_ab
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from srv import PY, Server, phases  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vjepa2")
ap.add_argument("--config", default=None)
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--cells", nargs="+", default=["1:20"])
ap.add_argument("--video", default="/root/work/mstar/test/qwen3-omni/video.webm")
ap.add_argument("--kwargs", default="{}")
ap.add_argument("--port", type=int, default=9110)
ap.add_argument("--gpus", default="0")
ap.add_argument("--period", default="10")
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--save", type=int, default=1)
ap.add_argument("--out", required=True)
args = ap.parse_args()

arms = dict(a.split("=", 1) for a in args.arms)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
config = args.config or f"configs/{args.model}.yaml"
summary = []
for rnd in range(args.rounds):
    order = list(arms) if rnd % 2 == 0 else list(arms)[::-1]
    for name in order:
        d = out / f"r{rnd}_{name}"
        log = str(d / "server.log")
        srv = Server(arms[name], [args.model, "--config", config], args.port, log, gpus=args.gpus,
                     env={"MSTAR_PHASE_TIMING": args.period})
        startup = srv.start()
        try:
            for cell in args.cells:
                c, n = map(int, cell.split(":"))
                cd = d / f"c{c}"
                r = subprocess.run([PY, str(Path(__file__).parent / "vjepa2_client.py"), "--port",
                                    str(args.port), "--video", args.video, "--n", str(n), "--c",
                                    str(c), "--warmup", str(args.warmup), "--kwargs", args.kwargs,
                                    "--save", str(args.save), "--out", str(cd)], capture_output=True, text=True)
                print(r.stdout[-2000:], r.stderr[-2000:])
                res = json.loads((cd / "results.json").read_text()) if (cd / "results.json").exists() else {}
                res.update(round=rnd, arm=name, startup_s=startup)
                summary.append(res)
                print(f"== r{rnd} {name} c={c}: {res.get('jct_p50_s')} s p50, {res.get('req_per_s')} req/s failed={res.get('failed')}", flush=True)
        finally:
            srv.stop()
        try:
            (d / "phases.txt").write_text(phases(log))
        except Exception as exc:  # noqa: BLE001
            (d / "phases.txt").write_text(f"phase parse failed: {exc}")
(out / "summary.json").write_text(json.dumps(summary, indent=1))
