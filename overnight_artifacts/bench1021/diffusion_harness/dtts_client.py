#!/usr/bin/env python3
"""Closed-loop streaming TTS client: one client and one policy for every engine.

Each request streams audio over HTTP; the client records
  TTFA   time from send to the first chunk that carries audio samples (container headers excluded)
  e2e    time to the last byte
  audio  seconds of audio decoded from the stream (PCM s16le at the engine's sample rate, or WAV)
  RTF    e2e / audio seconds
and per run: audio-s/s throughput = total audio seconds / wall time of the measured requests.

Engine request shapes live in REQUEST (engine -> builder) and are filled from each project's docs
(see dtts.py). Every response is content-checked: audio must be longer than 0.3 s, not silent
(RMS > 1e-3 of full scale), and not absurdly long (< 3 s per word + 2 s); a failed check is a failed
request. With --save-dir each measured request's audio is written as <i>.wav for the WER gate.

Request i (after warmup) speaks sentence (offset + i) % N with seed seed_base + i where the engine
takes a seed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def wav_header_len(buf: bytes) -> int | None:
    """Bytes before the PCM payload of a RIFF/WAV stream, or None if buf isn't WAV (yet)."""
    if len(buf) < 12 or buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return None
    i = 12
    while i + 8 <= len(buf):
        cid, size = buf[i:i + 4], struct.unpack("<I", buf[i + 4:i + 8])[0]
        if cid == b"data":
            return i + 8
        i += 8 + size
    return None


def wav_fmt(buf: bytes):
    """(channels, sample_rate, bytes_per_sample) from the fmt chunk (streamed headers often carry a 0 or
    0xFFFFFFFF data length, so the stdlib wave reader is not used)."""
    i = 12
    while i + 8 <= len(buf):
        cid, size = buf[i:i + 4], struct.unpack("<I", buf[i + 4:i + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, _, _, bits = struct.unpack("<HHIIHH", buf[i + 8:i + 24])
            return ch, sr, bits // 8
        i += 8 + size
    raise ValueError("no fmt chunk")


def decode_audio(body: bytes, fmt: str, sr: int) -> tuple[np.ndarray, int]:
    if fmt == "wav" or body[:4] == b"RIFF":
        hl = wav_header_len(body)
        if hl is not None:
            ch, sr, sw = wav_fmt(body)
            pcm = body[hl:]
            if sw == 2:
                x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
            elif sw == 4:
                x = np.frombuffer(pcm[: len(pcm) // 4 * 4], dtype="<f4")
            else:
                raise ValueError(f"sample width {sw}")
            return (x.reshape(-1, ch).mean(1) if ch > 1 else x), sr
    x = np.frombuffer(body[: len(body) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
    return x, sr


def write_wav(path: Path, x: np.ndarray, sr: int):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--spec", required=True, help="JSON: {path, body (with {text}/{seed} placeholders), "
                                                  "format: pcm|wav, sample_rate, form: false, headers}")
    ap.add_argument("--texts", required=True)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--warmup-waves", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(a.spec)
    texts = [t for t in Path(a.texts).read_text().splitlines() if t.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save = Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{a.port}{spec['path']}"

    def build(text, seed):
        body = json.loads(json.dumps(spec["body"]).replace('"{seed}"', str(seed)))
        def sub(v):
            if isinstance(v, str):
                return v.replace("{text}", text)
            if isinstance(v, dict):
                return {k: sub(x) for k, x in v.items()}
            if isinstance(v, list):
                return [sub(x) for x in v]
            return v
        return sub(body)

    def one(i, measured):
        idx = i if measured else 100000 + i
        k = (a.offset + idx) % len(texts)
        text, seed = texts[k], a.seed_base + idx
        rec = {"i": i, "text_index": k, "seed": seed, "start": time.time()}
        body = build(text, seed)
        if spec.get("form"):
            bnd = uuid.uuid4().hex
            data = b"".join(f'--{bnd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                            for k, v in body.items()) + f"--{bnd}--\r\n".encode()
            ctype = f"multipart/form-data; boundary={bnd}"
        else:
            data, ctype = json.dumps(body).encode(), "application/json"
        req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype, **spec.get("headers", {})})
        t0 = time.perf_counter()
        ttfa, buf = None, bytearray()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                while True:
                    chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    if ttfa is None:
                        hl = wav_header_len(bytes(buf)) if buf[:4] == b"RIFF" else 0
                        if hl is not None and len(buf) > hl + 64:
                            ttfa = time.perf_counter() - t0
            e2e = time.perf_counter() - t0
            x, sr = decode_audio(bytes(buf), spec.get("format", "pcm"), spec.get("sample_rate", 24000))
            dur = len(x) / sr
            rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
            words = len(text.split())
            reason = None
            if dur < 0.3:
                reason = f"audio too short ({dur:.2f}s)"
            elif rms < 1e-3:
                reason = f"silent (rms {rms:.2e})"
            elif dur > 3 * words + 2:
                reason = f"runaway length {dur:.1f}s for {words} words"
            rec.update(ok=reason is None, ttfa_s=ttfa, e2e_s=e2e, audio_s=dur, rtf=e2e / dur if dur else None,
                       rms=rms, sample_rate=sr, bytes=len(buf))
            if reason:
                rec["error"] = reason
            if save and measured and dur > 0:
                write_wav(save / f"{i:03d}.wav", x, sr)
        except urllib.error.HTTPError as e:
            rec.update(ok=False, e2e_s=time.perf_counter() - t0,
                       error=f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, e2e_s=time.perf_counter() - t0, error=repr(e)[:300])
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
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=a.conc) as pool:
            for f in [pool.submit(worker) for _ in range(a.conc)]:
                f.result()
        return sorted(recs, key=lambda r: r["i"]), time.perf_counter() - t0

    warm, _ = run(a.warmup_waves * a.conc, False) if a.warmup_waves else ([], 0)
    recs, wall = run(a.num, True)
    ok = [r for r in recs if r["ok"]]

    def pct(xs, p):
        xs = sorted(x for x in xs if x is not None)
        if not xs:
            return None
        k = (len(xs) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    audio = sum(r["audio_s"] for r in ok)
    summ = {"engine": a.engine, "conc": a.conc, "num": a.num, "completed": len(ok), "failed": len(recs) - len(ok),
            "warmup_failed": sum(not r["ok"] for r in warm), "wall_s": wall,
            "audio_s_per_s": audio / wall if wall else None, "audio_s_total": audio,
            "ttfa_p50": pct([r.get("ttfa_s") for r in ok], 50), "ttfa_p99": pct([r.get("ttfa_s") for r in ok], 99),
            "rtf_p50": pct([r.get("rtf") for r in ok], 50), "rtf_mean": statistics.fmean(r["rtf"] for r in ok)
            if ok else None, "e2e_p50": pct([r["e2e_s"] for r in ok], 50),
            "mean_audio_s": audio / len(ok) if ok else None,
            "errors": sorted({r.get("error", "")[:120] for r in recs if not r["ok"]})[:5]}
    (out / "results.json").write_text(json.dumps({"summary": summ, "args": vars(a), "requests": recs}, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
