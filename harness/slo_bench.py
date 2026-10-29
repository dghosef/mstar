"""Open-loop streaming TTS SLO client (Poisson arrivals), plus closed-loop collect/profile modes.

SLO per request (open-loop Poisson SLO methodology):
  * TTFA = first audio byte after the request was sent, must be <= 200 ms;
  * no stall: a player starts at the first byte and plays 24 kHz PCM16; every time the next
    bytes arrive after the player ran dry, the wait is a pause. Pauses summing >= 10 ms violate;
  * the request must finish (HTTP 200, audio, no error, within the timeout).
Attainment at a rate = SLO-met / sent (failures count as misses). Capacity at X% = the highest
rate whose attainment >= X and every lower rate's too.

Usage (on the box, clientenv python):
  slo_bench.py sweep  --engine mstar --url http://127.0.0.1:8100 --texts A.jsonl --rates 2,4,8 --out DIR
  slo_bench.py collect --engine mstar --url ... --texts T.jsonl --concurrency 4 --out DIR   (saves WAVs)
  slo_bench.py closed --engine mstar --url ... --texts T.jsonl --concurrency 32 --requests 256 --out DIR
  slo_bench.py report DIR...                                                              (capacity table)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR = 24000
BPS = SR * 2  # PCM16 mono bytes per second
TTFA_SLO = 0.200
STALL_SLO = 0.010
AUDIBLE = int(32768 * 10 ** (-45 / 20))  # -45 dBFS

SAMPLING = dict(temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05)
MAX_TOKENS = 2048


# Orpheus / Chatterbox: VoxServe's /v1/audio/speech takes no sampling fields and samples with its model
# defaults, so mstar is sent those same values (VoxServe's Chatterbox has no CFG: cfg_weight 0 matches it).
KIND_SAMPLING = {
    "orpheus": dict(temperature=0.6, top_p=0.8, repetition_penalty=1.3),
    "chatterbox": dict(temperature=0.8, top_p=0.95, repetition_penalty=1.2, cfg_weight=0.0, exaggeration=0.5),
}
KIND = "qwen3tts"


def payload(engine: str, text: str, voice: str, language: str, extra: dict) -> dict:
    if KIND != "qwen3tts":
        p = {"input": text, "voice": voice, "response_format": "pcm", "stream": True}
        if engine.startswith("mstar"):
            p.update(model=KIND, **KIND_SAMPLING[KIND], max_output_tokens=MAX_TOKENS)
        p.update(extra)
        return p
    if engine.startswith("mstar"):
        p = {"model": "qwen3_tts", "input": text, "voice": voice, "response_format": "pcm", "stream": True,
             "language": language, **SAMPLING, "max_output_tokens": MAX_TOKENS}
    elif engine.startswith("sgomni"):
        p = {"model": "qwen3-tts", "input": text, "voice": voice, "response_format": "pcm", "stream": True,
             "language": language, **SAMPLING, "max_new_tokens": MAX_TOKENS}
    elif engine.startswith("voxserve"):
        # sampling comes from the server CLI (its Qwen3-TTS defaults equal SAMPLING; --max-tokens 2048)
        p = {"input": text, "voice": voice, "language": language, "response_format": "pcm", "stream": True}
    else:
        raise ValueError(engine)
    p.update(extra)
    return p


def player_stall(events: list[tuple[float, int]]) -> tuple[float, int]:
    """Total pause (s) and pause count of a player that starts at the first byte."""
    if not events:
        return 0.0, 0
    start = events[0][0]
    played_bytes = 0
    stall = 0.0
    pauses = 0
    for t, n in events:
        due = start + stall + played_bytes / BPS
        if t > due:
            gap = t - due
            stall += gap
            if gap > 1e-4:
                pauses += 1
        played_bytes += n
    return stall, pauses


class Req:
    __slots__ = ("idx", "tid", "text", "t_sched", "t_send", "status", "err", "events", "t_end", "audible_t",
                 "bytes", "pcm")

    def __init__(self, idx, tid, text, t_sched):
        self.idx, self.tid, self.text, self.t_sched = idx, tid, text, t_sched
        self.t_send = self.t_end = self.audible_t = None
        self.status, self.err, self.events, self.bytes, self.pcm = None, None, [], 0, None

    def metrics(self) -> dict:
        ok = self.status == 200 and self.err is None and self.bytes > 0
        ttfa = (self.events[0][0] - self.t_send) if self.events else None
        stall, pauses = player_stall(self.events)
        audio_s = self.bytes / BPS
        e2e = (self.t_end - self.t_send) if self.t_end and self.t_send else None
        met = bool(ok and ttfa is not None and ttfa <= TTFA_SLO and stall < STALL_SLO)
        return {
            "idx": self.idx, "tid": self.tid, "status": self.status, "err": self.err, "ok": ok,
            "t_sched": self.t_sched, "t_send": self.t_send, "t_end": self.t_end, "ttfa": ttfa,
            "ttfa_audible": (self.audible_t - self.t_send) if self.audible_t else None,
            "stall": stall, "pauses": pauses, "audio_s": audio_s, "e2e": e2e, "met": met,
            "n_chunks": len(self.events),
            "events": [[round((t - self.t_send) * 1000, 2), n] for t, n in self.events],
        }


async def one(session, url, engine, r: Req, voice, language, extra, timeout, keep_pcm):
    import aiohttp
    r.t_send = time.perf_counter()
    buf = bytearray() if keep_pcm else None
    odd = b""  # a read may split a sample
    try:
        async with session.post(url, json=payload(engine, r.text, voice, language, extra),
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            r.status = resp.status
            if resp.status != 200:
                r.err = (await resp.text())[:300]
            else:
                async for data in resp.content.iter_any():
                    now = time.perf_counter()
                    if not data:
                        continue
                    r.events.append((now, len(data)))
                    r.bytes += len(data)
                    if buf is not None:
                        buf += data
                    if r.audible_t is None:
                        d = odd + data
                        cut = len(d) - (len(d) % 2)
                        odd = d[cut:]
                        if cut and np.abs(np.frombuffer(d[:cut], dtype="<i2").astype(np.int32)).max() > AUDIBLE:
                            r.audible_t = now
    except Exception as e:  # noqa: BLE001 - every failure is a recorded miss
        r.err = f"{type(e).__name__}: {e}"[:300]
    r.t_end = time.perf_counter()
    if buf is not None:
        r.pcm = bytes(buf[: len(buf) - len(buf) % 2])


def load_texts(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def save_wav(path: Path, pcm: bytes):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def summarize(ms: list[dict], rate: float | None, window: float | None) -> dict:
    n = len(ms)
    ok = [m for m in ms if m["ok"]]
    met = sum(m["met"] for m in ms)
    ttfa = np.array([m["ttfa"] for m in ok if m["ttfa"] is not None]) if ok else np.array([])
    aud = np.array([m["ttfa_audible"] for m in ok if m["ttfa_audible"] is not None]) if ok else np.array([])
    stall = np.array([m["stall"] for m in ok]) if ok else np.array([])
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else None  # noqa: E731
    span = None
    if ok:
        span = max(m["t_end"] for m in ok) - min(m["t_send"] for m in ms if m["t_send"])
    return {
        "rate": rate, "window_s": window, "sent": n, "ok": len(ok), "failed": n - len(ok), "met": met,
        "attainment": met / n if n else None,
        "ttfa_p50": pct(ttfa, 50), "ttfa_p90": pct(ttfa, 90), "ttfa_p99": pct(ttfa, 99),
        "ttfa_aud_p50": pct(aud, 50), "ttfa_aud_p99": pct(aud, 99),
        "ttfa_viol": int(sum(1 for m in ok if m["ttfa"] is None or m["ttfa"] > TTFA_SLO)),
        "stall_viol": int(sum(1 for m in ok if m["stall"] >= STALL_SLO)),
        "stall_p50": pct(stall, 50), "stall_p99": pct(stall, 99),
        "audio_s_mean": float(np.mean([m["audio_s"] for m in ok])) if ok else None,
        "e2e_p50": pct(np.array([m["e2e"] for m in ok]), 50) if ok else None,
        "completed_per_s": (len(ok) / span) if span else None,
        "goodput_per_s": (met / window) if window else None,
        "audio_s_per_s": (sum(m["audio_s"] for m in ok) / span) if span else None,
    }


def meta(args) -> dict:
    load = os.getloadavg()
    return {"engine": args.engine, "url": args.url, "texts": args.texts, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "loadavg": load, "label": getattr(args, "label", None)}


async def warmup(session, args, texts, extra):
    url = args.url.rstrip("/") + "/v1/audio/speech"
    for i in range(4):  # 4 warm-up requests
        r = Req(-1, texts[i % len(texts)]["id"], texts[i % len(texts)]["text"], 0)
        await one(session, url, args.engine, r, args.voice, args.language, extra, args.timeout, False)
        if r.status != 200 or r.err:
            raise SystemExit(f"warmup request failed: {r.status} {r.err}")
    # one untimed concurrent wave so batched shapes (graphs, compiles) are hit before timing
    wave_n = args.warm_wave
    rs = [Req(-1, texts[i % len(texts)]["id"], texts[i % len(texts)]["text"], 0) for i in range(wave_n)]
    await asyncio.gather(*(one(session, url, args.engine, r, args.voice, args.language, extra, args.timeout, False)
                           for r in rs))
    bad = [r for r in rs if r.status != 200 or r.err]
    print(f"warmup: 4 sequential + {wave_n} concurrent, {len(bad)} failed", flush=True)


async def run_rate(session, args, texts, extra, rate: float, seed: int) -> list[dict]:
    url = args.url.rstrip("/") + "/v1/audio/speech"
    rng = random.Random(seed * 1000003 + int(rate * 1000))
    t, sched = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t > args.window:
            break
        sched.append(t)
    order = list(range(len(texts)))
    if args.shuffle:
        rng.shuffle(order)
    reqs = [Req(i, texts[order[i % len(order)]]["id"], texts[order[i % len(order)]]["text"], s) for i, s in enumerate(sched)]
    t0 = time.perf_counter() + 0.5
    tasks = []
    for r in reqs:
        delay = t0 + r.t_sched - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(session, url, args.engine, r, args.voice, args.language, extra,
                                             args.timeout, False)))
    await asyncio.gather(*tasks)
    return [r.metrics() for r in reqs]


async def amain_sweep(args):
    import aiohttp
    texts = load_texts(args.texts)
    extra = json.loads(args.extra) if args.extra else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({**meta(args), "rates": args.rates, "window": args.window,
                                               "seed": args.seed, "extra": extra}, indent=1))
    conn = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(connector=conn) as session:
        if not args.no_warmup:
            await warmup(session, args, texts, extra)
        below = 0
        for rate in [float(x) for x in args.rates.split(",")]:
            f = out / f"rate_{rate:g}.jsonl"
            if f.exists() and not args.force:
                print(f"skip {f} (exists)")
                continue
            l0 = os.getloadavg()[0]
            ms = await run_rate(session, args, texts, extra, rate, args.seed)
            with open(f, "w") as fh:
                for m in ms:
                    fh.write(json.dumps(m) + "\n")
            s = summarize(ms, rate, args.window)
            s["load_before"], s["load_after"] = l0, os.getloadavg()[0]
            with open(out / "summary.jsonl", "a") as fh:
                fh.write(json.dumps(s) + "\n")
            print(f"rate {rate:g}: sent {s['sent']} ok {s['ok']} attain {s['attainment']:.3f} "
                  f"ttfa p50 {fmt(s['ttfa_p50'])} p99 {fmt(s['ttfa_p99'])} ttfa_viol {s['ttfa_viol']} "
                  f"stall_viol {s['stall_viol']} done/s {fmt(s['completed_per_s'])} goodput {fmt(s['goodput_per_s'])} "
                  f"load {l0:.1f}", flush=True)
            below = below + 1 if (s["attainment"] or 0) < args.stop_below else 0
            if below >= 2:
                print("two consecutive rates below the stop floor; ending the sweep", flush=True)
                break
            await asyncio.sleep(args.pause)


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


async def amain_closed(args, save_audio: bool):
    import aiohttp
    texts = load_texts(args.texts)
    extra = json.loads(args.extra) if args.extra else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/") + "/v1/audio/speech"
    n = args.requests or len(texts)
    reqs = [Req(i, texts[i % len(texts)]["id"], texts[i % len(texts)]["text"], 0) for i in range(n)]
    sem = asyncio.Semaphore(args.concurrency)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        if not args.no_warmup:
            await warmup(session, args, texts, extra)

        async def run(r):
            async with sem:
                await one(session, url, args.engine, r, args.voice, args.language, extra, args.timeout, save_audio)
        t0 = time.perf_counter()
        await asyncio.gather(*(run(r) for r in reqs))
        wall = time.perf_counter() - t0
    ms = [r.metrics() for r in reqs]
    if save_audio:
        (out / "wav").mkdir(exist_ok=True)
        for r in reqs:
            if r.pcm:
                save_wav(out / "wav" / f"{r.idx:04d}_{r.tid}.wav", r.pcm)
    with open(out / "requests.jsonl", "w") as fh:
        for m in ms:
            fh.write(json.dumps({**m, "text": reqs[m["idx"]].text}) + "\n")
    s = summarize(ms, None, None)
    s.update(meta(args), concurrency=args.concurrency, wall_s=wall)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(f"closed c={args.concurrency}: n {n} ok {s['ok']} failed {s['failed']} wall {wall:.1f}s "
          f"audio-s/s {sum(m['audio_s'] for m in ms if m['ok']) / wall:.1f} ttfa p50 {fmt(s['ttfa_p50'])} "
          f"p99 {fmt(s['ttfa_p99'])}", flush=True)


def capacity(summ: list[dict], q: float) -> float | None:
    best = None
    for s in sorted(summ, key=lambda s: s["rate"]):
        if (s["attainment"] or 0) >= q:
            best = s["rate"]
        else:
            break
    return best


def report(dirs: list[str]):
    print("| run | cap@90 | cap@99 | max done/s | max goodput/s | idle TTFA p50 (ms) | rates |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for d in dirs:
        p = Path(d) / "summary.jsonl"
        if not p.exists():
            continue
        summ = {}
        for line in open(p):
            s = json.loads(line)
            summ[s["rate"]] = s  # a rerun of a rate replaces it
        summ = sorted(summ.values(), key=lambda s: s["rate"])
        idle = [s["ttfa_p50"] for s in summ[:2] if s["ttfa_p50"] is not None]
        rates = " ".join(f"{s['rate']:g}:{s['attainment']:.2f}" for s in summ)
        print(f"| {d} | {capacity(summ, 0.9)} | {capacity(summ, 0.99)} | "
              f"{max((s['completed_per_s'] or 0) for s in summ):.1f} | {max((s['goodput_per_s'] or 0) for s in summ):.1f} | "
              f"{1000 * min(idle):.0f} | {rates} |" if idle else f"| {d} | - | - | - | - | - | {rates} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "collect", "closed", "report"])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--engine", default="mstar")
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--texts")
    ap.add_argument("--rates", default="2,4,8")
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="shuffle text order per rate (workload B)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--pause", type=float, default=5.0)
    ap.add_argument("--stop-below", type=float, default=0.3)
    ap.add_argument("--voice", default="ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--extra", default="")
    ap.add_argument("--warm-wave", type=int, default=16)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--kind", default="qwen3tts", choices=["qwen3tts", "orpheus", "chatterbox"])
    args = ap.parse_args()
    global KIND
    KIND = args.kind
    if args.mode == "report":
        report(args.dirs)
        return
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    if args.mode == "sweep":
        asyncio.run(amain_sweep(args))
    else:
        asyncio.run(amain_closed(args, save_audio=args.mode == "collect"))


if __name__ == "__main__":
    sys.exit(main())
