"""Where does a vjepa2 request's client-side time go? Start a server (arm) and time phases."""
import base64
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from srv import Server  # noqa: E402

arm, port, video = sys.argv[1], int(sys.argv[2]), sys.argv[3]
proto = sys.argv[4] if len(sys.argv) > 4 else "ndjson"
gpu = sys.argv[5] if len(sys.argv) > 5 else "1"
srv = Server(arm, ["vjepa2", "--config", "configs/vjepa2.yaml"], port, f"/workspace/results/world/vjepa2_probe/server_{proto}.log",
             gpus=gpu)
srv.start()
try:
    data_bytes = Path(video).read_bytes()
    for i in range(6):
        t0 = time.perf_counter()
        files = [("files", (Path(video).name, data_bytes, "video/webm"))]
        data = {"input_modalities": "video", "output_modalities": "video", "model_kwargs": "{}"}
        headers = {"Accept": "application/vnd.mstar.frames"} if proto == "binary" else {}
        r = requests.post(f"http://127.0.0.1:{port}/generate", data=data, files=files, stream=True, headers=headers)
        t_hdr = time.perf_counter()
        buf = b"".join(r.iter_content(chunk_size=1 << 20))
        t_body = time.perf_counter()
        n = 0
        if proto == "binary":
            pos = 0
            while pos < len(buf):
                nl = buf.index(b"\n", pos)
                hdr = json.loads(buf[pos:nl])
                k = int(hdr.get("nbytes") or 0)
                n += k
                pos = nl + 1 + k
        else:
            for line in buf.split(b"\n"):
                if line.startswith(b"{") and b'"data"' in line:
                    n += len(base64.b64decode(json.loads(line)["data"]))
        t_parse = time.perf_counter()
        print(json.dumps({"i": i, "headers_s": t_hdr - t0, "body_s": t_body - t_hdr, "parse_s": t_parse - t_body,
                          "total_s": t_parse - t0, "bytes": len(buf), "payload": n, "ctype": r.headers.get("content-type")}),
              flush=True)
finally:
    srv.stop()
