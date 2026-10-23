"""Greedy correctness + targeted latency client for Qwen3.5 on /generate.

    python qcheck.py --url URL --set /root/work/imgset --out DIR [--conc 1] [--max-tokens 64]
        [--repeat 1] [--only big|small|text]

The set directory holds manifest.json: [{"id", "prompt", "image" (path or null)}].
Writes results.json: per id the text, ttft and e2e latency; prints a summary.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import aiohttp


async def one(session, url, item, max_tokens, extra):
    form = aiohttp.FormData()
    form.add_field("text", item["prompt"])
    kw = {"temperature": 0.0, "max_tokens": max_tokens, "max_output_tokens": max_tokens,
          "enable_thinking": False, "seed": 0, **extra}
    form.add_field("model_kwargs", json.dumps(kw))
    form.add_field("output_modalities", "text")
    if item.get("image"):
        form.add_field("input_modalities", "image,text")
        p = Path(item["image"])
        form.add_field("files", p.read_bytes(), filename=p.name, content_type="application/octet-stream")
    t0 = time.perf_counter()
    first = None
    chunks = []
    async with session.post(f"{url}/generate", data=form, read_bufsize=2**24) as resp:
        if resp.status != 200:
            return {"id": item["id"], "error": f"HTTP {resp.status} {await resp.text()}"}
        async for raw in resp.content:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not msg.get("data"):
                continue
            first = first or time.perf_counter() - t0
            chunks.append(base64.b64decode(msg["data"]))
    e2e = time.perf_counter() - t0
    return {"id": item["id"], "text": b"".join(chunks).decode("utf-8", "replace"),
            "ttft": first, "e2e": e2e}


async def main_async(a):
    items = json.loads((Path(a.set) / "manifest.json").read_text())
    if a.only:
        items = [it for it in items if it.get("kind") == a.only]
    items = items * a.repeat
    sem = asyncio.Semaphore(a.conc)
    extra = json.loads(a.extra) if a.extra else {}

    async def guarded(s, it):
        async with sem:
            return await one(s, a.url, it, a.max_tokens, extra)

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=1800)) as s:
        # warmup: every item once, unmeasured
        for it in items[: len(items) // a.repeat]:
            await one(s, a.url, it, 4, extra)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(guarded(s, it) for it in items))
        wall = time.perf_counter() - t0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    errs = [r for r in res if "error" in r]
    ok = [r for r in res if "error" not in r]
    summ = {
        "n": len(res), "errors": len(errs), "wall_s": wall, "req_per_s": len(ok) / wall,
        "ttft_p50": statistics.median(r["ttft"] for r in ok) if ok else None,
        "ttft_mean": statistics.mean(r["ttft"] for r in ok) if ok else None,
        "e2e_p50": statistics.median(r["e2e"] for r in ok) if ok else None,
    }
    by_kind = {}
    kinds = {it["id"]: it.get("kind") for it in items}
    for r in ok:
        by_kind.setdefault(kinds[r["id"]], []).append(r["ttft"])
    summ["ttft_p50_by_kind"] = {k: statistics.median(v) for k, v in by_kind.items()}
    (out / "results.json").write_text(json.dumps({"summary": summ, "rows": res, "errors": errs}, indent=1))
    print(json.dumps(summ, indent=1))
    for e in errs[:3]:
        print("ERR", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default=None)
    ap.add_argument("--extra", default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
