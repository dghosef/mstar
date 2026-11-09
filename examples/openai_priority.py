"""Priority scheduling: urgent requests jump a queue of background ones.

Serve with the priority admission policy, and a KV pool small enough that requests queue:

    # priority.yaml: the model's config plus
    admission_policy: priority
    resources:
      kv_cache:
        max_num_pages: 24

    pip install openai
    mstar serve qwen3_5_0.8b --config priority.yaml
    python examples/openai_priority.py --model qwen3_5_0.8b

It sends a burst of background requests at priority 10, then a few urgent ones at priority 0, and prints the order
in which the requests got their first token. Under ``admission_policy: priority`` the urgent requests start ahead of
the background ones still queued. Under ``fcfs`` they wait their turn. ``priority`` goes in the request body
(``extra_body`` from the OpenAI SDK): an integer, lower first, 0 when unset.
"""

import argparse
import asyncio
import time

from openai import AsyncOpenAI


async def one(client: AsyncOpenAI, model: str, name: str, priority: int, prompt: str, tokens: int, t0: float) -> dict:
    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=tokens, stream=True,
        # ignore_eos keeps every reply at ``tokens``, so the requests differ only in priority
        extra_body={"priority": priority, "ignore_eos": True},
    )
    first = None
    async for chunk in stream:
        if first is None and chunk.choices and chunk.choices[0].delta.content is not None:
            first = time.perf_counter() - t0
    return {"name": name, "priority": priority, "first": first, "done": time.perf_counter() - t0}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen3_5_0.8b")
    parser.add_argument("--background", type=int, default=24)
    parser.add_argument("--urgent", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between the burst and the urgent requests")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="none")
    prompt = " ".join(f"word{i}" for i in range(200)) + ". Summarize."
    t0 = time.perf_counter()
    tasks = [
        asyncio.create_task(one(client, args.model, f"bg{i}", 10, f"{i} {prompt}", args.tokens, t0))
        for i in range(args.background)
    ]
    await asyncio.sleep(args.delay)
    tasks += [
        asyncio.create_task(one(client, args.model, f"urgent{i}", 0, f"u{i} {prompt}", args.tokens, t0))
        for i in range(args.urgent)
    ]
    results = await asyncio.gather(*tasks)

    started = sorted(results, key=lambda r: r["first"])
    print("first-token order:", " ".join(r["name"] for r in started))
    bg_after = [sum(o["first"] > r["first"] for o in results if o["priority"] == 10)
                for r in results if r["priority"] == 0]
    print(f"each urgent request started ahead of this many background requests: {bg_after}")


if __name__ == "__main__":
    asyncio.run(main())
