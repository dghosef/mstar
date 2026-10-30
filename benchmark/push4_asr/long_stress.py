"""Long-file burst against a transcription server: N concurrent requests of a W-minute file (the 10 min long_form
LibriSpeech file looped to W minutes, each request starting at a different offset), each must finish with HTTP 200
and a non-trivial transcript in --timeout.
The case a Qwen3-ASR conductor cap guards: long prompts filling the KV pool.

  python long_stress.py --url http://127.0.0.1:28200 --model qwen3_asr --n 16 --minutes 20 --out stress.json
"""
import argparse
import asyncio
import io
import json
import time
import wave
from pathlib import Path

import aiohttp


def long_wav(minutes, shift):
    """``minutes`` of the long-form file, looped, starting ``shift`` seconds in: different requests carry different
    audio, so no prefix-cache page is shared between them."""
    src = next((Path.home() / "push4/asr/data/long_form").glob("*.wav"))
    with wave.open(str(src), "rb") as r:
        params, frames = r.getparams(), r.readframes(r.getnframes())
    need = int(minutes * 60 * params.framerate) * params.sampwidth
    off = (int(shift * params.framerate) * params.sampwidth) % len(frames)
    data = ((frames[off:] + frames[:off]) * (need // len(frames) + 1))[:need]
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(data)
    return out.getvalue()


async def one(s, a, audio, i):
    form = aiohttp.FormData()
    for k, v in (("model", a.model), ("response_format", "json"), ("temperature", "0"), ("language", "en")):
        form.add_field(k, v)
    form.add_field("file", audio, filename=f"long{i}.wav", content_type="audio/wav")
    t0 = time.perf_counter()
    try:
        async with s.post(f"{a.url}/v1/audio/transcriptions", data=form) as r:
            body = await r.text()
            text = json.loads(body).get("text", "") if r.status == 200 else body[:200]
            return {"i": i, "status": r.status, "secs": round(time.perf_counter() - t0, 1), "words": len(text.split())}
    except Exception as e:  # noqa: BLE001
        return {"i": i, "status": None, "secs": round(time.perf_counter() - t0, 1), "error": repr(e)[:200]}


async def main(a):
    audios = [long_wav(a.minutes, 7.3 * i) for i in range(a.n)]
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=a.timeout)) as s:
        res = await asyncio.gather(*(one(s, a, audios[i], i) for i in range(a.n)))
    ok = [r for r in res if r["status"] == 200 and r["words"] > 50 * a.minutes]
    summary = {"n": a.n, "minutes": a.minutes, "ok": len(ok), "wall_s": round(time.perf_counter() - t0, 1),
               "max_secs": max(r["secs"] for r in res), "words_min": min((r.get("words") or 0) for r in res)}
    print(json.dumps(summary), flush=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "requests": res}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
