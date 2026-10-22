"""Robustness probe for a running wan22 sp2 server: mixed guidance concurrently, a client abort
mid-request, then a final request that must still complete (no lockstep hang).

python sp2_robust.py <port>
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

port = int(sys.argv[1])


def gen(guidance, seed, timeout, steps=8, size="832x480"):
    body = json.dumps({"prompt": "a bicycle leaning against a tree", "size": size, "seed": seed,
                       "guidance_scale": guidance, "num_inference_steps": steps, "num_frames": 33}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/videos/generations", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
        return f"ok gs={guidance} {time.perf_counter() - t0:.1f}s bytes={len(out['data'][0]['b64_json'])}"
    except Exception as e:  # noqa: BLE001
        return f"ERR gs={guidance} after {time.perf_counter() - t0:.1f}s: {type(e).__name__}"


with ThreadPoolExecutor(5) as ex:
    futs = [ex.submit(gen, g, i, 300) for i, g in enumerate([5.0, 1.0, 5.0, 1.0])]
    futs.append(ex.submit(gen, 5.0, 99, 1.5))  # aborted by the client mid-request
    for f in futs:
        print(f.result(), flush=True)
print("final:", gen(5.0, 7, 300), flush=True)
print("final no-cfg:", gen(1.0, 8, 300), flush=True)
