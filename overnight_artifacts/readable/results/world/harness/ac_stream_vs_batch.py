"""Single request: streaming walk (per-iter emit) vs accumulated walk; print per-chunk sums and diffs."""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import requests

port, video, kwp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
vid = Path(video).read_bytes()
kw = json.loads(Path(kwp).read_text()) if kwp.endswith(".json") else json.loads(kwp)
D = int(sys.argv[4]) if len(sys.argv) > 4 else 1408
T = int(sys.argv[5]) if len(sys.argv) > 5 else 256


def one(k):
    files = [("files", ("v.mp4", vid, "video/mp4"))]
    data = {"input_modalities": "video", "output_modalities": "video", "model_kwargs": json.dumps(k)}
    out = []
    with requests.post(f"http://127.0.0.1:{port}/generate", data=data, files=files, stream=True) as r:
        for line in r.iter_lines(chunk_size=1 << 20):
            if line:
                m = json.loads(line)
                if m.get("modality") == "video" and m.get("data"):
                    out.append(np.frombuffer(base64.b64decode(m["data"]), dtype=np.float32))
    return out


acc = one(kw)
st = one(dict(kw, stream_rollout=True))
print("accumulated: messages", len(acc), [a.size for a in acc])
print("streaming: messages", len(st), [a.size for a in st])
C = T * D
A = np.concatenate(acc).reshape(-1, C)
S = np.concatenate(st).reshape(-1, C)
print("acc chunk sums", [f"{x.astype(np.float64).sum():.6f}" for x in A])
print("stream chunk sums", [f"{x.astype(np.float64).sum():.6f}" for x in S])
print("acc[i] vs stream[j] rel:", [[round(float(np.linalg.norm(A[i] - S[j]) / np.linalg.norm(S[j])), 3) for j in range(len(S))] for i in range(len(A))])
