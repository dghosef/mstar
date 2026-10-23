"""Shared closed-loop load generator for every arm (mstar, vLLM, SGLang): one client,
one client policy.

OpenAI /v1/chat/completions, stream=True, temperature 0, ignore_eos with a fixed
max_tokens per request (taken from the dataset row, so request i does the same work on
every engine), thinking off (flat `enable_thinking` for mstar, `chat_template_kwargs`
for vLLM/SGLang; both are sent to every engine).

Metrics (the same definitions as `vllm bench serve`):
  TTFT  first SSE chunk carrying non-empty content/reasoning, from the send
  TPOT  (e2e - TTFT) / (output_len - 1), output_len = the forced max_tokens
  ITL   gaps between content-bearing chunks (an engine that coalesces tokens into one
        chunk shows fewer, larger gaps; TPOT is the comparable number)
  output tok/s = sum of output tokens of completed requests / wall time of the measured
        closed loop (warmup excluded). Output tokens are the engine's usage
        completion_tokens when it reports them (vLLM/SGLang), else the forced max_tokens
        (mstar reports zeros); the per-request check below makes sure that is honest.

A request counts as failed if: HTTP != 200, a transport error or timeout, an SSE error
event, no content, or (when the engine reports usage) completion_tokens != max_tokens.
For engines without usage, the re-tokenized length of the streamed text and the count
of content chunks are recorded so the work can be audited.

  python client.py --url http://127.0.0.1:8300 --model-id Qwen/Qwen3.5-0.8B \
      --data data/random_1024_256.jsonl --conc 32 --num 160 --warmup-waves 5 --out cell/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import resource

import aiohttp

try:  # the client must not be the bottleneck at tens of thousands of SSE events/s
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads
try:
    import uvloop
except ImportError:
    uvloop = None


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


OPTS = {"extra": {}, "count_actual": False, "save_text": False}
_MEDIA: dict[str, str] = {}
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".mp4": "video/mp4",
         ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg"}


def media_url(path: str) -> str:
    """Media travel as data URLs (every engine accepts them), encoded once per file."""
    if path not in _MEDIA:
        import base64
        mime = _MIME.get(Path(path).suffix.lower(), "application/octet-stream")
        _MEDIA[path] = f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()
    return _MEDIA[path]


def body_for(model_id: str, row: dict, ignore_eos: bool) -> dict:
    content = row["prompt"]
    parts = [{"type": t, t: {"url": media_url(row[k])}} for k, t in
             (("image", "image_url"), ("video", "video_url"), ("audio", "audio_url")) if row.get(k)]
    if parts:
        content = parts + [{"type": "text", "text": row["prompt"]}]
    body = {
        "model": model_id,
        "messages": ([{"role": "system", "content": row["system"]}] if row.get("system") else [])
                    + [{"role": "user", "content": content}],
        "max_tokens": int(row["output_len"]),
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body.update(OPTS["extra"])
    return body


async def one(session: aiohttp.ClientSession, url: str, model_id: str, row: dict, ignore_eos: bool,
              timeout: float) -> dict:
    rec = {"id": row["id"], "output_len": int(row["output_len"]), "input_len": row.get("input_len"),
           "ok": False, "error": None, "ttft": None, "e2e": None, "chunks": 0, "usage_tokens": None,
           "finish_reason": None, "text_chars": 0}
    gaps: list[float] = []
    text_parts: list[str] = []
    t0 = time.perf_counter()
    last = None
    try:
        async with session.post(url, json=body_for(model_id, row, ignore_eos),
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                rec["error"] = f"HTTP {r.status}: {(await r.text())[:300]}"
                return rec
            buf = b""
            async for raw in r.content.iter_any():
                buf += raw
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    for line in ev.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            continue
                        now = time.perf_counter()
                        obj = _loads(data)
                        if "error" in obj:
                            rec["error"] = f"SSE error: {str(obj['error'])[:300]}"
                            continue
                        u = obj.get("usage")
                        if u and u.get("completion_tokens"):
                            rec["usage_tokens"] = int(u["completion_tokens"])
                        for ch in obj.get("choices") or []:
                            d = ch.get("delta") or {}
                            piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                            if piece:
                                if rec["ttft"] is None:
                                    rec["ttft"] = now - t0
                                else:
                                    gaps.append(now - last)
                                last = now
                                rec["chunks"] += 1
                                text_parts.append(piece)
                            if ch.get("finish_reason"):
                                rec["finish_reason"] = ch["finish_reason"]
        rec["e2e"] = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 -- a failed request is a result, not a crash
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        return rec
    text = "".join(text_parts)
    rec["text_chars"] = len(text)
    rec["text_head"] = text[:120]
    rec["_text"] = text
    rec["_gaps"] = gaps
    if OPTS["save_text"]:
        rec["text"] = text
    if rec["error"] is None:
        if not text:
            rec["error"] = "empty output"
        elif (not OPTS["count_actual"] and rec["usage_tokens"] is not None
              and rec["usage_tokens"] != rec["output_len"]):
            rec["error"] = f"usage completion_tokens {rec['usage_tokens']} != max_tokens {rec['output_len']}"
        else:
            rec["ok"] = True
    if rec["ttft"] is not None and rec["e2e"] is not None and rec["output_len"] > 1:
        rec["tpot"] = (rec["e2e"] - rec["ttft"]) / (rec["output_len"] - 1)
    return rec


async def closed_loop(url, model_id, rows, conc, ignore_eos, timeout):
    sem = asyncio.Semaphore(conc)
    conn = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(connector=conn) as s:
        async def go(row):
            async with sem:
                return await one(s, url, model_id, row, ignore_eos, timeout)
        t0 = time.perf_counter()
        recs = await asyncio.gather(*(go(r) for r in rows))
        wall = time.perf_counter() - t0
    return recs, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data", required=True, help="jsonl rows {id, prompt, output_len, input_len}")
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--warmup-waves", type=int, default=1)
    ap.add_argument("--warmup-offset", type=int, default=-1, help="rows for warmup start here (default: after the measured rows)")
    ap.add_argument("--no-ignore-eos", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tokenizer", default=None, help="re-tokenize streamed text to audit engines without usage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-body", default="{}", help="JSON merged into every request body")
    ap.add_argument("--count-actual", action="store_true",
                    help="real EOS allowed: count output tokens as usage (else re-tokenized text), "
                         "not the requested max_tokens")
    ap.add_argument("--save-text", action="store_true", help="keep each reply's full text in results.json")
    a = ap.parse_args()
    OPTS.update(extra=json.loads(a.extra_body), count_actual=a.count_actual, save_text=a.save_text)
    rows = [json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
    meas = [rows[(a.offset + i) % len(rows)] for i in range(a.num)]
    woff = a.warmup_offset if a.warmup_offset >= 0 else a.offset + a.num
    warm = [rows[(woff + i) % len(rows)] for i in range(a.warmup_waves * a.conc)]
    url = a.url.rstrip("/") + "/v1/chat/completions"
    ignore_eos = not a.no_ignore_eos
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if warm:
        wrecs, wwall = asyncio.run(closed_loop(url, a.model_id, warm, a.conc, ignore_eos, a.timeout))
        wfail = sum(not r["ok"] for r in wrecs)
        print(f"warmup: {len(warm)} requests in {wwall:.1f}s, failed {wfail}", flush=True)
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    recs, wall = asyncio.run(closed_loop(url, a.model_id, meas, a.conc, ignore_eos, a.timeout))
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    client_cpu = (ru1.ru_utime - ru0.ru_utime + ru1.ru_stime - ru0.ru_stime) / wall if wall else None

    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    gaps_all: list[float] = []
    for r in recs:
        text = r.pop("_text", "")
        gaps_all += r.pop("_gaps", [])
        if tok is not None and text:
            r["retok_tokens"] = len(tok(text, add_special_tokens=False)["input_ids"])
        if OPTS["count_actual"]:
            r["n_out"] = r["usage_tokens"] or r.get("retok_tokens") or r["chunks"]
            if r.get("ttft") is not None and r.get("e2e") is not None and r["n_out"] > 1:
                r["tpot"] = (r["e2e"] - r["ttft"]) / (r["n_out"] - 1)
    ok = [r for r in recs if r["ok"]]
    failed = len(recs) - len(ok)
    out_tokens = sum((r.get("n_out") if OPTS["count_actual"] else (r["usage_tokens"] or r["output_len"])) or 0
                     for r in ok)
    in_tokens = sum((r.get("input_len") or 0) for r in ok)
    ttft = [r["ttft"] for r in ok]
    tpot = [r["tpot"] for r in ok if r.get("tpot") is not None]
    e2e = [r["e2e"] for r in ok]
    summ = {
        "completed": len(ok), "failed": failed, "num": len(recs), "conc": a.conc, "wall_s": wall,
        "output_tok_s": out_tokens / wall if wall else None,
        "total_tok_s": (out_tokens + in_tokens) / wall if wall else None,
        "req_s": len(ok) / wall if wall else None,
        "mean_output_len": statistics.mean([(r.get("n_out") if OPTS["count_actual"] else r["output_len"]) or 0
                                            for r in ok]) if ok else None,
        "mean_input_len": statistics.mean([(r.get("input_len") or 0) for r in ok]) if ok else None,
        "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99), "ttft_mean": statistics.mean(ttft) if ttft else None,
        "tpot_p50": pct(tpot, 50), "tpot_p99": pct(tpot, 99), "tpot_mean": statistics.mean(tpot) if tpot else None,
        "itl_p50": pct(gaps_all, 50), "itl_p99": pct(gaps_all, 99),
        "e2e_p50": pct(e2e, 50), "e2e_p99": pct(e2e, 99),
        "chunks_per_token": (sum(r["chunks"] for r in ok) / sum(r["output_len"] for r in ok)) if ok else None,
        "usage_reported": sum(r["usage_tokens"] is not None for r in ok),
        "retok_ratio": (sum(r.get("retok_tokens", 0) for r in ok) / sum(r["output_len"] for r in ok))
        if ok and tok is not None else None,
        "client_cpu_frac": client_cpu, "json": _loads.__module__, "uvloop": uvloop is not None,
        "errors": sorted({r["error"] for r in recs if r["error"]})[:10],
    }
    (out / "results.json").write_text(json.dumps({"summary": summ, "requests": recs}, indent=0))
    def ms(x):
        return f"{x * 1000:.1f}" if x is not None else "-"
    print(f"RESULT conc={a.conc} n={len(recs)} failed={failed} wall={wall:.2f}s out_tok/s={summ['output_tok_s']:.1f} "
          f"ttft p50/p99={ms(summ['ttft_p50'])}/{ms(summ['ttft_p99'])}ms tpot p50/p99={ms(summ['tpot_p50'])}/"
          f"{ms(summ['tpot_p99'])}ms chunks/tok={summ['chunks_per_token']} retok={summ['retok_ratio']} "
          f"client_cpu={client_cpu:.2f} errors={summ['errors'][:2]}", flush=True)


if __name__ == "__main__":
    if uvloop is not None:
        uvloop.install()
    main()
