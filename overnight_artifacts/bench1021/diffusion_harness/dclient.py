#!/usr/bin/env python3
"""Closed-loop diffusion client: one client and one client policy for every served engine.

Engines and routes (each engine's own OpenAI-style surface; same prompts, size, steps,
guidance, seeds on all):
  image  mstar / vllmomni / sglang: POST /v1/images/generations (JSON, b64_json PNG)
  video  mstar:    POST /v1/videos/generations (JSON, data[0].b64_json mp4)
         vllmomni: POST /v1/videos/sync (multipart form, raw mp4 body)
         sglang:   POST /v1/videos (JSON) -> poll GET /v1/videos/{id} (0.1 s) -> GET .../content

Request i (0-based, counted after warmup) uses prompt (offset + i) % len(prompts) and seed
seed_base + i, so request i is the same work on every engine. `generator_device: cpu` is
sent to the competitors that accept it per request (images on vllm-omni and SGLang, video
on SGLang), matching mstar's and the diffusers reference's CPU-generator noise.

Every response is content-checked: an image must decode to the requested size and not be
flat (pixel std > 2); a video must decode to the requested size and frame count. A request
that fails either check counts as failed, like an HTTP error.

Closed loop: `conc` workers, each issuing its next request as soon as its previous one
returns, until `num` measured requests are done. `warmup_waves` x conc untimed requests go
first, with seeds/prompts disjoint from the measured ones.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def _post_json(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _post_form(url, fields, timeout):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(url, data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


class Spec:
    def __init__(self, a):
        self.a = a
        self.w, self.h = (int(x) for x in a.size.split("x"))

    def request(self, prompt: str, seed: int) -> bytes:
        a, base = self.a, f"http://127.0.0.1:{self.a.port}"
        if a.kind == "image":
            body = {"model": a.model_id, "prompt": prompt, "size": a.size, "n": 1, "response_format": "b64_json",
                    "num_inference_steps": a.steps, "guidance_scale": a.guidance, "seed": seed}
            if a.engine in ("vllmomni", "sglang"):
                body["generator_device"] = "cpu"
            if a.engine == "sglang":
                body.update(width=self.w, height=self.h)
            payload = json.loads(_post_json(f"{base}/v1/images/generations", body, a.timeout))
            return base64.b64decode(payload["data"][0]["b64_json"])
        common = {"prompt": prompt, "size": a.size, "num_frames": a.frames, "fps": a.fps, "seed": seed,
                  "num_inference_steps": a.steps, "guidance_scale": a.guidance, "negative_prompt": a.negative_prompt}
        if a.flow_shift is not None:
            common["flow_shift"] = a.flow_shift
        if a.engine == "mstar":
            body = {"model": a.model_id, "response_format": "b64_json", **common}
            body.pop("flow_shift", None)  # mstar reads the checkpoint's scheduler config
            payload = json.loads(_post_json(f"{base}/v1/videos/generations", body, a.timeout))
            return base64.b64decode(payload["data"][0]["b64_json"])
        if a.engine == "vllmomni":
            fields = {"model": a.model_id, "width": self.w, "height": self.h, **common}
            fields.pop("size")
            return _post_form(f"{base}/v1/videos/sync", fields, a.timeout)
        if a.engine == "sglang":
            body = {"model": a.model_id, "width": self.w, "height": self.h, "generator_device": "cpu", **common}
            job = json.loads(_post_json(f"{base}/v1/videos", body, a.timeout))
            t0 = time.time()
            while True:
                st = json.loads(_get(f"{base}/v1/videos/{job['id']}", 60))
                if st.get("status") == "completed":
                    break
                if st.get("status") in ("failed", "cancelled", "error"):
                    raise RuntimeError(f"sglang video job {st.get('status')}: {st.get('error')}")
                if time.time() - t0 > a.timeout:
                    raise TimeoutError("sglang video job timed out")
                time.sleep(0.1)
            return _get(f"{base}/v1/videos/{job['id']}/content", 600)
        raise ValueError(a.engine)

    def check(self, data: bytes) -> str | None:
        """None when the output is valid, else the reason."""
        if not data:
            return "empty body"
        if self.a.kind == "image":
            try:
                im = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:  # noqa: BLE001
                return f"undecodable image: {e!r}"
            if im.size != (self.w, self.h):
                return f"size {im.size} != {(self.w, self.h)}"
            if float(np.asarray(im).std()) <= 2.0:
                return "flat image"
            return None
        try:
            import av
            with av.open(io.BytesIO(data)) as c:
                frames = [f for f in c.decode(video=0)]
            n, (w, h) = len(frames), (frames[0].width, frames[0].height)
        except Exception as e:  # noqa: BLE001
            return f"undecodable video: {e!r}"
        if (w, h) != (self.w, self.h):
            return f"video size {(w, h)} != {(self.w, self.h)}"
        if n != self.a.frames:
            return f"frames {n} != {self.a.frames}"
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["mstar", "vllmomni", "sglang"])
    ap.add_argument("--kind", required=True, choices=["image", "video"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--size", required=True, help="WxH")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--guidance", type=float, required=True)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--flow-shift", type=float, default=None)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--warmup-waves", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--prompt-offset", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--save-dir", default=None, help="save every measured output as <i>.png/.mp4")
    ap.add_argument("--out", required=True, help="cell directory (results.json)")
    a = ap.parse_args()

    prompts = [p for p in Path(a.prompts).read_text().splitlines() if p.strip()]
    spec = Spec(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save = Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    def one(i: int, measured: bool) -> dict:
        idx = i if measured else 100000 + i
        prompt = prompts[(a.prompt_offset + idx) % len(prompts)]
        seed = a.seed_base + idx
        t_start = time.time()
        t0 = time.perf_counter()
        rec = {"i": i, "seed": seed, "prompt_index": (a.prompt_offset + idx) % len(prompts), "start": t_start}
        try:
            data = spec.request(prompt, seed)
            rec["latency_s"] = time.perf_counter() - t0
            reason = spec.check(data)
            rec["ok"] = reason is None
            if reason:
                rec["error"] = reason
            rec["bytes"] = len(data)
            if save and measured and data:
                (save / f"{i:03d}.{'png' if a.kind == 'image' else 'mp4'}").write_bytes(data)
        except urllib.error.HTTPError as e:
            rec.update(ok=False, latency_s=time.perf_counter() - t0,
                       error=f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, latency_s=time.perf_counter() - t0, error=repr(e)[:300])
        rec["end"] = time.time()
        return rec

    def run(n: int, measured: bool) -> tuple[list[dict], float]:
        lock, nxt, recs = threading.Lock(), [0], []

        def worker():
            while True:
                with lock:
                    if nxt[0] >= n:
                        return
                    i = nxt[0]
                    nxt[0] += 1
                r = one(i, measured)
                with lock:
                    recs.append(r)
                    if not r["ok"]:
                        print(f"  !! req {i} failed: {r.get('error')}", flush=True)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=a.conc) as pool:
            for f in [pool.submit(worker) for _ in range(a.conc)]:
                f.result()
        return sorted(recs, key=lambda r: r["i"]), time.perf_counter() - t0

    warm, warm_s = run(a.warmup_waves * a.conc, measured=False) if a.warmup_waves else ([], 0.0)
    recs, wall = run(a.num, measured=True)
    ok = [r for r in recs if r["ok"]]
    lat = sorted(r["latency_s"] for r in ok)

    def pct(p):
        if not lat:
            return None
        k = (len(lat) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(lat) - 1)
        return lat[lo] + (lat[hi] - lat[lo]) * (k - lo)

    summary = {
        "engine": a.engine, "kind": a.kind, "conc": a.conc, "num": a.num, "completed": len(ok),
        "failed": len(recs) - len(ok), "warmup_requests": len(warm), "warmup_failed": sum(not r["ok"] for r in warm),
        "warmup_s": round(warm_s, 2), "wall_s": round(wall, 3),
        "throughput_per_s": len(ok) / wall if wall else None,
        "throughput_per_min": 60 * len(ok) / wall if wall else None,
        "lat_mean": statistics.fmean(lat) if lat else None, "lat_p50": pct(50), "lat_p90": pct(90),
        "lat_p99": pct(99), "lat_min": lat[0] if lat else None, "lat_max": lat[-1] if lat else None,
        "errors": sorted({r.get("error", "")[:120] for r in recs if not r["ok"]})[:5],
    }
    args = {k: v for k, v in vars(a).items()}
    (out / "results.json").write_text(json.dumps({"summary": summary, "args": args, "requests": recs,
                                                 "warmup": warm}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
