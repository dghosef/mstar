"""Shared driver for dgen's offline competitor arms (the official package's own batch inference, run in
its own venv). Contract with dgen.offline_run:

  python -I <script> --workload WL --items ITEMS.jsonl --batch-sizes 1,16,64 --nums 64,256,512 --warmup 1
         --seed-base S --out DIR [--save-dir DIR] [model-specific args]

For each batch size b: `warmup` untimed batches, then `num` requests (request i = item i % N, seed S + i)
in batches of b, timed end to end per batch (input decode included, as a server would). Writes
DIR/results.json {"rows": [{batch, num, completed, failed, wall_s, throughput_per_s, units_per_s,
lat_mean, lat_p50, lat_p99 (per-batch latency, every request of a batch gets it), errors}]} and, for
text outputs, DIR/outputs.jsonl ({i, item, text}); binary outputs go to --save-dir/<i>.<ext>.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path


def parse(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--batch-sizes", required=True)
    ap.add_argument("--nums", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-dir", default=None)
    if extra:
        extra(ap)
    a = ap.parse_args()
    a.items_list = [json.loads(x) for x in Path(a.items).read_text().splitlines() if x.strip()]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.save_dir:
        Path(a.save_dir).mkdir(parents=True, exist_ok=True)
    return a


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run(a, process, sync=None, order=None, ext="bin"):
    """process(list of (i, item, seed)) -> list of outputs (str for text, bytes for binary, or Exception;
    or (output, units) to count units measured on the output, e.g. the audio seconds of a WAV).
    order(list of i) -> the order requests are batched in (e.g. length-sorted); default: as given."""
    items = a.items_list
    rows, texts = [], []
    for b, n in zip([int(x) for x in a.batch_sizes.split(",")], [int(x) for x in a.nums.split(",")]):
        reqs = [(i, i % len(items), a.seed_base + i) for i in range(n)]
        if order:
            reqs = order(reqs)
        err = []
        try:
            for w in range(a.warmup):
                wr = [(100000 + w * b + j, (len(items) - 1 - j) % len(items), 900000 + j) for j in range(b)]
                process([(i, items[k], s) for i, k, s in wr])
            if sync:
                sync()
            lat, done, failed, units, rtf = [], 0, 0, 0.0, []
            t0 = time.perf_counter()
            for s in range(0, n, b):
                chunk = reqs[s:s + b]
                t1 = time.perf_counter()
                try:
                    outs = process([(i, items[k], sd) for i, k, sd in chunk])
                    if sync:
                        sync()
                except Exception as e:  # noqa: BLE001
                    outs = [e] * len(chunk)
                dt = time.perf_counter() - t1
                for (i, k, _), o in zip(chunk, outs):
                    if isinstance(o, Exception) or o is None:
                        failed += 1
                        err.append(repr(o)[:200])
                        continue
                    if isinstance(o, tuple):  # (output, units measured on it, e.g. audio seconds)
                        o, u = o
                        rtf.append(dt / u if u else None)
                    else:
                        u = items[k].get("units", 1)
                    done += 1
                    units += u
                    lat.append(dt)
                    if isinstance(o, str):
                        texts.append({"i": i, "item": k, "text": o})
                    elif a.save_dir and isinstance(o, (bytes, bytearray)):
                        (Path(a.save_dir) / f"{i:03d}.{ext}").write_bytes(o)
            wall = time.perf_counter() - t0
            rows.append({"batch": b, "num": n, "completed": done, "failed": failed, "wall_s": round(wall, 3),
                         "throughput_per_s": done / wall if wall else None,
                         "throughput_per_min": 60 * done / wall if wall else None,
                         "units_per_s": units / wall if wall else None, "units_total": units,
                         "lat_mean": statistics.fmean(lat) if lat else None, "lat_p50": pct(lat, 50),
                         "lat_p90": pct(lat, 90), "lat_p99": pct(lat, 99), "ttfb_p50": None, "ttfb_p99": None,
                         "errors": sorted(set(err))[:5]})
            if rtf:  # TTS: audio-s/s, RTF (batch latency / own audio), TTFA = batch latency (no streaming)
                rr = [x for x in rtf if x is not None]
                rows[-1].update(audio_s_per_s=rows[-1]["units_per_s"], audio_s_total=units, rtf_p50=pct(rr, 50),
                                rtf_mean=statistics.fmean(rr) if rr else None, ttfa_p50=pct(lat, 50),
                                ttfa_p99=pct(lat, 99), e2e_p50=pct(lat, 50),
                                mean_audio_s=units / done if done else None)
        except Exception as e:  # noqa: BLE001
            rows.append({"batch": b, "num": n, "completed": 0, "failed": None,
                         "error": traceback.format_exc()[-1500:]})
            print(f"batch {b} crashed: {e!r}", flush=True)
        print(json.dumps(rows[-1])[:600], flush=True)
        (Path(a.out) / "results.json").write_text(json.dumps({"rows": rows}, indent=1))
    if texts:
        (Path(a.out) / "outputs.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in sorted(texts, key=lambda t: t["i"])))
    (Path(a.out) / "results.json").write_text(json.dumps({"rows": rows}, indent=1))
