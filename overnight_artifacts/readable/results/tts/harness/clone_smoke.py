"""OmniVoice clone smoke: N clone requests (ref_audio data URL + ref_text) at concurrency C.

    python clone_smoke.py --port 8100 --ref REF.wav --ref-text "..." --out DIR
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path

import aiohttp

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Please call me back at your earliest convenience, I have some important news to share.",
    "Turn left on Maple Avenue, continue for half a mile, then take the second right onto Cedar Lane.",
    "The meeting has been rescheduled to Thursday at two thirty in the afternoon.",
]


async def one(session, url, body, path):
    async with session.post(url, json=body) as r:
        data = await r.read()
        if r.status != 200:
            return {"status": r.status, "err": data[:300].decode(errors="replace")}
        path.write_bytes(data)
        return {"status": 200, "bytes": len(data)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = "data:audio/wav;base64," + base64.b64encode(Path(a.ref).read_bytes()).decode()
    url = f"http://127.0.0.1:{a.port}/v1/audio/speech"
    lines = []
    tasks = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        for i in range(a.repeat * len(TEXTS)):
            text = TEXTS[i % len(TEXTS)]
            lines.append(text)
            body = {"model": "omnivoice", "input": text, "response_format": "wav",
                    "ref_audio": ref, "ref_text": a.ref_text}
            tasks.append(one(s, url, body, out / f"{i + 1:04d}.wav"))
        res = await asyncio.gather(*tasks)
    (out.parent / f"{out.name}_sentences.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"ok": sum(r["status"] == 200 for r in res), "n": len(res),
                      "errors": [r for r in res if r["status"] != 200][:3]}))


asyncio.run(main())
