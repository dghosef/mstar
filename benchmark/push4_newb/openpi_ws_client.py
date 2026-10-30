"""openpi's own serving path for pi0.5: a client of scripts/serve_policy.py (websocket, msgpack) on the same DROID
episodes as run_pi05.py, so openpi pays for serving as mstar does over HTTP. Runs in the openpi env.

  openpi/.venv/bin/python openpi_ws_client.py --port P --n 50 --warm 5 --local-cache DIR --out DIR

Per request it records two times: `infer_ms` (WebsocketClientPolicy.infer, round trip with numpy images) and
`obs_infer_ms` (also decoding the camera images from disk, as the mstar server does with the uploaded files).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: benchmark.dataset, benchmark.openpi_pi05
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

from benchmark.dataset import DROIDDataset  # noqa: E402
from benchmark.openpi_pi05 import _build_obs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--local-cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reqs = DROIDDataset(local_file_dir=a.local_cache, num_requests=a.n, task="pi05").get_requests()
    policy = WebsocketClientPolicy(host="127.0.0.1", port=a.port)
    for i in range(a.warm):
        policy.infer(_build_obs(reqs[i % len(reqs)]))
    per, failed = [], 0
    for i, req in enumerate(reqs):
        try:
            t0 = time.perf_counter()
            obs = _build_obs(req)
            t1 = time.perf_counter()
            actions = np.asarray(policy.infer(obs)["actions"])
            t2 = time.perf_counter()
            per.append({"request_id": i, "infer_ms": (t2 - t1) * 1000, "obs_infer_ms": (t2 - t0) * 1000,
                        "jct_ms": (t2 - t1) * 1000, "finite": bool(np.isfinite(actions).all()),
                        "shape": list(actions.shape)})
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"req {i}: FAILED {e!r}", flush=True)
    (out / "results.json").write_text(json.dumps({"failed": failed, "per_request": per}, indent=1))
    print(f"done: {len(per)} ok, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
