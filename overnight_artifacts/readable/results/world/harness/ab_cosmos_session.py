"""One strict request/response Cosmos3 DROID action session: ab_cosmos_session.py PORT ITERS"""
import statistics
import sys
import time

import msgpack
import websockets.sync.client

port, iters = int(sys.argv[1]), int(sys.argv[2])
obs = open("/root/work/media/droid_obs.jpg", "rb").read()
mk = {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10, "action_chunk_size": 32,
      "num_frames": 33}
lat = []
with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/generate/ws", max_size=None) as ws:
    for i in range(iters):
        t = time.perf_counter()
        ws.send(msgpack.packb({"text": "Pick up the red cup and place it in the sink.",
                               "files": [{"name": "o.jpg", "data": obs}], "input_modalities": ["image", "text"],
                               "output_modalities": ["action"], "model_kwargs": mk, "request_id": f"s{t}-{i}"},
                              use_bin_type=True))
        while True:
            r = msgpack.unpackb(ws.recv(), raw=False)
            if "error" in r:
                raise RuntimeError(r["error"])
            if r.get("finish"):
                break
        lat.append(time.perf_counter() - t)
print("median ms", 1e3 * statistics.median(lat[2:] if len(lat) > 3 else lat))
