"""Probe vjepa2_ac batched-rollout row independence on a running server.

Waves of 4 simultaneous requests: (1) four identical requests A, (2) A,A,A,B with B's
actions scaled. Saves every output. Row independence means: within a wave of identical
requests all outputs are equal, and A's outputs don't change when one partner is B.

  python ac_batch_probe.py PORT VIDEO KWARGS_JSON OUTDIR [waves]
"""
import base64
import json
import sys
import threading
from pathlib import Path

import numpy as np
import requests

port, video, kw_path, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
waves = int(sys.argv[5]) if len(sys.argv) > 5 else 3
out.mkdir(parents=True, exist_ok=True)
vid = Path(video).read_bytes()
kw_a = json.loads(Path(kw_path).read_text())
kw_b = dict(kw_a, actions=[[2.0 * v for v in row] for row in kw_a["actions"]])


def one(kw):
    files = [("files", ("v.mp4", vid, "video/mp4"))]
    data = {"input_modalities": "video", "output_modalities": "video", "model_kwargs": json.dumps(kw)}
    chunks = []
    with requests.post(f"http://127.0.0.1:{port}/generate", data=data, files=files, stream=True, timeout=600) as r:
        for line in r.iter_lines(chunk_size=1 << 20):
            if line:
                m = json.loads(line)
                if m.get("modality") == "video" and m.get("data"):
                    chunks.append(base64.b64decode(m["data"]))
    return np.frombuffer(b"".join(chunks), dtype=np.float32)


def wave(kws):
    res = [None] * len(kws)
    barrier = threading.Barrier(len(kws))

    def run(i):
        barrier.wait()
        res[i] = one(kws[i])
    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(kws))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return res


def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(a.astype(np.float64)))


single_a = one(kw_a)
single_a2 = one(kw_a)
single_b = one(kw_b)
print("single A vs A (determinism):", rel(single_a, single_a2))
report = {"single_A_vs_A": rel(single_a, single_a2)}
for w in range(waves):
    same = wave([kw_a] * 4)
    mixed = wave([kw_a, kw_a, kw_a, kw_b])
    r = {
        "same_rows_vs_row0": [rel(same[0], x) for x in same],
        "same_row0_vs_single": rel(single_a, same[0]),
        "mixed_A_rows_vs_same_row0": [rel(same[0], x) for x in mixed[:3]],
        "mixed_B_vs_single_B": rel(single_b, mixed[3]),
    }
    print(w, json.dumps(r))
    report[f"wave{w}"] = r
    np.save(out / f"same_w{w}.npy", np.stack(same))
    np.save(out / f"mixed_w{w}.npy", np.stack(mixed))
(out / "report.json").write_text(json.dumps(report, indent=1))
