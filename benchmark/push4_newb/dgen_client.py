#!/usr/bin/env python3
"""Closed-loop HTTP client for the dgen entries (ASR, BAGEL, video / world models): one client and one
policy for every served engine, driven by a request spec (JSON) and an items file (JSONL).

Request i (0-based, after warmup) uses item (offset + i) % N and seed seed_base + i, so request i is the
same work on every engine. Placeholders in the spec's body / files / query: a string that is exactly
"{name}" becomes the item's value (any JSON type; "{seed}" the seed), and "{name}" inside a longer string
is replaced by str(value).

Spec keys:
  path       URL path (may carry placeholders)
  encoding   json | multipart | none (GET)
  body       request body template (json: any JSON; multipart: flat dict of form fields)
  files      multipart file fields {field: "{audio_path}"} (path on disk)
  response   json | bytes | stream | sse | mstar_frames (M* /generate with Accept:
             application/vnd.mstar.frames: header line + raw bytes per chunk; payloads concatenated)
  extract    for json: dotted path to the output ("text", "data.0.b64_json", "choices.0.message.content");
             for sse: dotted path inside each event's JSON ("choices.0.delta.content"); texts are joined
  decode     b64: base64-decode the extracted string
  check      text | image | video | bytes | json   (a failed check is a failed request)
  save       txt | png | mp4 | bin | json          (measured outputs, with --save-dir)
  save_tail  keep only the last N bytes of a binary output (e.g. the last frame chunk of a long rollout)
  poll       optional {"status_path": "/v1/videos/{id}", "content_path": "/v1/videos/{id}/content",
             "id_key": "id", "done": "completed"} for job-style APIs

Per request: latency (send -> last byte), TTFB (send -> first body byte; first text delta for sse).
Per run: requests/s, units/s (sum of the items' "units" over completed requests / wall: audio seconds
for ASR, i.e. RTFx; frames for video; 1 per request by default), latency mean/p50/p90/p99, TTFB p50/p99.
Text outputs of measured requests go to outputs.jsonl (for the gate); binary ones to --save-dir.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def fill(v, ctx):
    if isinstance(v, str):
        if v.startswith("{") and v.endswith("}") and v[1:-1] in ctx and v.count("{") == 1:
            return ctx[v[1:-1]]
        for k, x in ctx.items():
            if "{" + k + "}" in v:
                v = v.replace("{" + k + "}", str(x))
        return v
    if isinstance(v, dict):
        return {k: fill(x, ctx) for k, x in v.items()}
    if isinstance(v, list):
        return [fill(x, ctx) for x in v]
    return v


def dig(obj, path):
    for p in path.split("."):
        if p == "":
            continue
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    return obj


def multipart(fields, files):
    bnd = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        parts.append(f'--{bnd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, p in files.items():
        paths = p if isinstance(p, list) else [p]
        for one in paths:
            data = Path(one).read_bytes()
            ctype = mimetypes.guess_type(str(one))[0] or "application/octet-stream"
            parts.append(f'--{bnd}\r\nContent-Disposition: form-data; name="{k}"; filename="{Path(one).name}"\r\n'
                         f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{bnd}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={bnd}"


def check(kind, out):
    """None when the output is valid, else the reason."""
    if kind == "text":
        return None if isinstance(out, str) and out.strip() else "empty text"
    if kind == "json":
        return None if out is not None else "no json"
    if not out:
        return "empty body"
    if kind == "image":
        import numpy as np
        from PIL import Image
        try:
            im = Image.open(io.BytesIO(out)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            return f"undecodable image: {e!r}"
        return "flat image" if float(np.asarray(im).std()) <= 2.0 else None
    if kind == "video":
        try:
            import av
            with av.open(io.BytesIO(out)) as c:
                n = sum(1 for _ in c.decode(video=0))
        except Exception as e:  # noqa: BLE001
            return f"undecodable video: {e!r}"
        return None if n > 0 else "no frames"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--spec", required=True, help="JSON string or @file")
    ap.add_argument("--items", required=True, help="JSONL; optional 'units' per item")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--warmup-waves", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec[1:]).read_text() if a.spec.startswith("@") else a.spec)
    items = [json.loads(x) for x in Path(a.items).read_text().splitlines() if x.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save = Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{a.port}"
    texts_lock, texts = threading.Lock(), []

    def http(method_path, data=None, ctype=None, timeout=None):
        hdrs = dict(spec.get("headers", {}))
        if ctype:
            hdrs["Content-Type"] = ctype
        req = urllib.request.Request(base + method_path, data=data, headers=hdrs)
        return urllib.request.urlopen(req, timeout=timeout or a.timeout)

    def one(i, measured):
        idx = i if measured else 100000 + i
        k = (a.offset + idx) % len(items)
        item = items[k]
        ctx = {**item, "seed": a.seed_base + idx, "i": idx}
        rec = {"i": i, "item": k, "seed": ctx["seed"], "start": time.time(), "units": item.get("units", 1)}
        t0 = time.perf_counter()
        ttfb = None
        try:
            path = fill(spec["path"], ctx)
            enc = spec.get("encoding", "json")
            if enc == "json":
                data, ctype = json.dumps(fill(spec.get("body", {}), ctx)).encode(), "application/json"
            elif enc == "multipart":
                data, ctype = multipart(fill(spec.get("body", {}), ctx), fill(spec.get("files", {}), ctx))
            else:
                data, ctype = None, None
            resp_kind = spec.get("response", "json")
            with http(path, data, ctype) as r:
                if resp_kind == "sse":
                    deltas, pending = [], b""
                    while True:
                        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                        if not chunk:
                            break
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            line = line.strip()
                            if not line.startswith(b"data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == b"[DONE]":
                                continue
                            try:
                                d = dig(json.loads(payload), spec.get("extract", "choices.0.delta.content"))
                            except (ValueError, KeyError, IndexError, TypeError):
                                continue
                            if d:
                                if ttfb is None:
                                    ttfb = time.perf_counter() - t0
                                deltas.append(d)
                    result = "".join(deltas)
                else:
                    buf = bytearray()
                    while True:
                        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                        if not chunk:
                            break
                        if ttfb is None:
                            ttfb = time.perf_counter() - t0
                        buf += chunk
                    body = bytes(buf)
                    result = body
                    if resp_kind == "mstar_frames":  # M* /generate binary framing: {json header}\n + nbytes
                        payload, pos, nchunks = bytearray(), 0, 0
                        while pos < len(body):
                            nl = body.index(b"\n", pos)
                            hdr = json.loads(body[pos:nl])
                            if "error" in hdr:
                                raise RuntimeError(f"stream error: {hdr}")
                            n = int(hdr.get("nbytes", 0))
                            payload += body[nl + 1:nl + 1 + n]
                            pos, nchunks = nl + 1 + n, nchunks + 1
                        if nchunks < spec.get("min_chunks", 1):
                            raise RuntimeError(f"{nchunks} chunks < {spec.get('min_chunks', 1)}")
                        rec["chunks"] = nchunks
                        result = bytes(payload)
                    elif resp_kind == "json":
                        obj = json.loads(body)
                        if spec.get("poll"):
                            pl = spec["poll"]
                            jid = dig(obj, pl.get("id_key", "id"))
                            while True:
                                with http(pl["status_path"].replace("{id}", str(jid)), timeout=60) as rs:
                                    st = json.loads(rs.read())
                                s = str(dig(st, pl.get("status_key", "status")))
                                if s == pl.get("done", "completed"):
                                    break
                                if s in ("failed", "error", "cancelled"):
                                    raise RuntimeError(f"job {s}: {st}")
                                if time.perf_counter() - t0 > a.timeout:
                                    raise TimeoutError("job timed out")
                                time.sleep(0.1)
                            with http(pl["content_path"].replace("{id}", str(jid)), timeout=600) as rc:
                                obj = rc.read()
                            result = obj
                        else:
                            result = dig(obj, spec["extract"]) if spec.get("extract") else obj
                        if spec.get("decode") == "b64" and isinstance(result, str):
                            result = base64.b64decode(result.split(",", 1)[1] if result.startswith("data:") else result)
            rec["latency_s"] = time.perf_counter() - t0
            rec["ttfb_s"] = ttfb
            reason = check(spec.get("check", "bytes"), result)
            rec["ok"] = reason is None
            if reason:
                rec["error"] = reason
            if measured:
                if isinstance(result, str):
                    with texts_lock:
                        texts.append({"i": i, "item": k, "text": result})
                elif isinstance(result, (bytes, bytearray)):
                    rec["bytes"] = len(result)
                    if save and result:
                        keep = result[-spec["save_tail"]:] if spec.get("save_tail") else result
                        (save / f"{i:03d}.{spec.get('save', 'bin')}").write_bytes(keep)
                elif save and result is not None:
                    (save / f"{i:03d}.json").write_text(json.dumps(result)[:2_000_000])
        except urllib.error.HTTPError as e:
            rec.update(ok=False, latency_s=time.perf_counter() - t0,
                       error=f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, latency_s=time.perf_counter() - t0, error=repr(e)[:300])
        rec["end"] = time.time()
        return rec

    def run(n, measured):
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

    warm, warm_s = run(a.warmup_waves * a.conc, False) if a.warmup_waves else ([], 0.0)
    recs, wall = run(a.num, True)
    ok = [r for r in recs if r["ok"]]
    lat = sorted(r["latency_s"] for r in ok)
    tfb = sorted(r["ttfb_s"] for r in ok if r.get("ttfb_s") is not None)

    def pct(xs, p):
        if not xs:
            return None
        k = (len(xs) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    units = sum(r["units"] for r in ok)
    summary = {
        "engine": a.engine, "conc": a.conc, "num": a.num, "completed": len(ok), "failed": len(recs) - len(ok),
        "warmup_requests": len(warm), "warmup_failed": sum(not r["ok"] for r in warm), "warmup_s": round(warm_s, 2),
        "wall_s": round(wall, 3), "throughput_per_s": len(ok) / wall if wall else None,
        "throughput_per_min": 60 * len(ok) / wall if wall else None,
        "units_per_s": units / wall if wall else None, "units_total": units,
        "lat_mean": statistics.fmean(lat) if lat else None, "lat_p50": pct(lat, 50), "lat_p90": pct(lat, 90),
        "lat_p99": pct(lat, 99), "lat_min": lat[0] if lat else None, "lat_max": lat[-1] if lat else None,
        "ttfb_p50": pct(tfb, 50), "ttfb_p99": pct(tfb, 99),
        "errors": sorted({r.get("error", "")[:120] for r in recs if not r["ok"]})[:5],
    }
    if texts:
        (out / "outputs.jsonl").write_text("".join(json.dumps(t) + "\n" for t in sorted(texts, key=lambda t: t["i"])))
    (out / "results.json").write_text(json.dumps({"summary": summary, "args": vars(a), "requests": recs,
                                                 "warmup": warm}, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
