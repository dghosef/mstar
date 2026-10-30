"""GSM8K accuracy over the OpenAI chat API, the same client for every engine (L5).

openai/gsm8k main test split, first N questions; 5-shot as multi-turn chat (the first
5 train examples, answers in their '#### n' form); greedy, thinking off, max_tokens 512,
real EOS (no ignore_eos). Score = last number after '####' (else the last number in the
reply) equals the reference.

  python gsm8k.py --url http://127.0.0.1:8300 --model-id Qwen/Qwen3.5-0.8B --n 250 --conc 32 --out gsm8k_<arm>.json
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp
from datasets import load_dataset

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract(text: str):
    if "####" in text:
        nums = NUM.findall(text.split("####")[-1])
        if nums:
            return nums[0].replace(",", "").rstrip(".")
    nums = NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def norm(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


async def main_async(a):
    ds = load_dataset("openai/gsm8k", "main")
    shots = ds["train"].select(range(5))
    test = ds["test"].select(range(a.n))
    inst = "Solve the math problem step by step. End your answer with a line '#### <number>'."
    fewshot = []
    for s in shots:
        fewshot += [{"role": "user", "content": f"{inst}\n\n{s['question']}"},
                    {"role": "assistant", "content": s["answer"]}]
    sem = asyncio.Semaphore(a.conc)
    url = a.url.rstrip("/") + "/v1/chat/completions"

    async def one(sess, i, q):
        body = {"model": a.model_id, "messages": fewshot + [{"role": "user", "content": f"{inst}\n\n{q['question']}"}],
                "max_tokens": 512, "temperature": 0.0, "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False}, **json.loads(a.extra_body)}
        async with sem:
            try:
                async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=1200)) as r:
                    js = await r.json(content_type=None)
                msg = js["choices"][0]["message"]
                text = (msg.get("content") or "")
            except Exception as e:  # noqa: BLE001
                return {"i": i, "error": repr(e)[:200], "correct": False}
        ref = q["answer"].split("####")[-1].strip().replace(",", "")
        pred = extract(text)
        ok = norm(pred) is not None and norm(ref) is not None and abs(norm(pred) - norm(ref)) < 1e-6
        return {"i": i, "pred": pred, "ref": ref, "correct": ok, "reply": text[-300:]}

    async with aiohttp.ClientSession() as sess:
        res = await asyncio.gather(*(one(sess, i, q) for i, q in enumerate(test)))
    acc = sum(r["correct"] for r in res) / len(res)
    errs = sum("error" in r for r in res)
    Path(a.out).write_text(json.dumps({"accuracy": acc, "n": len(res), "errors": errs, "rows": res}, indent=0))
    print(f"GSM8K n={len(res)} accuracy={acc:.4f} errors={errs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-body", default="{}", help="JSON merged into every request (e.g. Omni text-only)")
    asyncio.run(main_async(ap.parse_args()))
