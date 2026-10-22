"""Lockstep stress for a running video server (TP/SP instance): rounds of concurrent requests with
mixed guidance and steps, some cancelled by the client mid-flight, then probe requests that must
finish (a TP/SP desync shows up as the probe hanging).

python lockstep_stress.py <port> <WxH> <frames> [rounds]
"""
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

port, size, frames = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
rng = random.Random(0)


def gen(guidance, seed, steps, timeout):
    body = json.dumps({"prompt": "a bicycle leaning against a tree in the rain", "size": size, "seed": seed,
                       "guidance_scale": guidance, "num_inference_steps": steps, "num_frames": frames}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/videos/generations", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.load(r)
        return True, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__} after {time.perf_counter() - t0:.1f}s"


ok = cancelled = bad = 0
for rnd in range(rounds):
    jobs = []
    with ThreadPoolExecutor(6) as ex:
        for i in range(6):
            cancel = i % 3 == 2
            g = rng.choice([1.0, 5.0, 6.0])
            steps = rng.choice([4, 6, 8])
            timeout = rng.uniform(0.5, 3.0) if cancel else 600
            jobs.append((cancel, ex.submit(gen, g, rnd * 10 + i, steps, timeout)))
        for cancel, f in jobs:
            good, info = f.result()
            if good:
                ok += 1
            elif cancel:
                cancelled += 1
            else:
                bad += 1
                print("unexpected failure:", info, flush=True)
    print(f"round {rnd}: ok={ok} cancelled={cancelled} bad={bad}", flush=True)
for g in (6.0, 1.0):
    good, info = gen(g, 777, 4, 300)
    print(f"probe gs={g}: {'ok %.1fs' % info if good else 'FAIL ' + info}", flush=True)
    bad += not good
print("STRESS", "PASS" if bad == 0 else "FAIL", flush=True)
