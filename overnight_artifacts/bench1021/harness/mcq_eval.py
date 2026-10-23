"""MMMU-style accuracy over the OpenAI chat API, the same client for every engine.

Rows from vision_data.py's mmmu.jsonl (single image + question + lettered options +
"Answer with the option's letter ..."); greedy, thinking off, max_tokens 16, real EOS.
Score: the first standalone option letter in the reply equals the answer.

  python -I mcq_eval.py --url http://127.0.0.1:8300 --model-id Qwen/Qwen3.5-9B --rows mmmu.jsonl --out mmmu_<tag>.json
"""
import argparse
import asyncio
import base64
import json
import re
from pathlib import Path

import aiohttp

LET = re.compile(r"\b([A-J])\b")


def data_url(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


async def main_async(a):
    rows = [json.loads(x) for x in Path(a.rows).read_text().splitlines() if x.strip()]
    sem = asyncio.Semaphore(a.conc)
    url = a.url.rstrip("/") + "/v1/chat/completions"
    extra = json.loads(a.extra_body)

    async def one(sess, r):
        body = {"model": a.model_id, "max_tokens": 16, "temperature": 0.0, "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url(r["image"])}},
                    {"type": "text", "text": r["prompt"]}]}], **extra}
        async with sem:
            try:
                async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=1200)) as resp:
                    js = await resp.json(content_type=None)
                text = js["choices"][0]["message"].get("content") or ""
            except Exception as e:  # noqa: BLE001
                return {"id": r["id"], "error": repr(e)[:200], "correct": False}
        m = LET.search(text.strip().upper()[:40])
        pred = m.group(1) if m else None
        return {"id": r["id"], "pred": pred, "answer": r["answer"], "correct": pred == r["answer"], "reply": text[:200]}

    async with aiohttp.ClientSession() as sess:
        res = await asyncio.gather(*(one(sess, r) for r in rows))
    acc = sum(x["correct"] for x in res) / len(res)
    errs = sum("error" in x for x in res)
    Path(a.out).write_text(json.dumps({"accuracy": acc, "n": len(res), "errors": errs, "rows": res}, indent=0))
    print(f"MMMU n={len(res)} accuracy={acc:.4f} errors={errs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))
