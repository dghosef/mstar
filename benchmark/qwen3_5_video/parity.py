"""Greedy video answers from a served Qwen3.8-27B, teacher-forced through transformers.

The video counterpart of ``benchmark/qwen3_8/parity.py`` (same method).
``collect`` asks a running server (mstar, vLLM or SGLang; OpenAI chat with
``video_url`` data URLs) for greedy completions of a fixed set of video
prompts. ``oracle`` loads the checkpoint in transformers, builds each prompt
with the HF processor from the same file (torchcodec decode, 2 fps sampling,
timestamps), and teacher-forces every collected completion: at each position,
is the served token transformers' argmax, and if not, how far below the
argmax its logit sits. Greedy text is not reproducible across engines or
server processes, so the yardstick is the agreement of the other engines and
of repeated rounds of one engine.

Thinking is off on every side (``enable_thinking`` for mstar,
``chat_template_kwargs`` for vLLM / SGLang, the template kwarg for
transformers), so answers are about the video rather than a long preamble.

  python parity.py collect --port 8300 --tag mstar_round1 --data DIR --out-dir results/parity
  python parity.py oracle --data DIR --out-dir results/parity
"""
import argparse
import base64
import json
import urllib.request
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.8-27B"
MAX_TOKENS = 128
IMAGE = ("nvidia/Cosmos3-Edge", "assets/example_action_fd_umi_first_frame.png")


def prompts(data: Path) -> list[dict]:
    """Four whole videos with their Video-MME question, two short clips, and
    one prompt mixing an image and a video."""
    long = [json.loads(line) for line in open(data / "long/data/items.jsonl")]
    short = [json.loads(line) for line in open(data / "short/data/items.jsonl")]
    out = []
    for row in long[:4]:
        opts = "\n".join(row["options"])
        out.append({"video": str(data / "long" / row["video_path"]),
                    "text": f"{row['question']}\n{opts}\nAnswer with the option's letter, then one sentence why."})
    for row in short[:2]:
        out.append({"video": str(data / "short" / row["video_path"]),
                    "text": "Describe what happens in this clip in two sentences."})
    out.append({"image": True, "video": str(data / "short" / short[2]["video_path"]),
                "text": "Does the image show anything that also appears in the video? Answer briefly."})
    return out


def _image_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(IMAGE[0], IMAGE[1])


def _data_url(path: str, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def collect(args):
    url = f"http://localhost:{args.port}/v1/chat/completions"
    results = []
    for i, p in enumerate(prompts(Path(args.data))):
        content = []
        if p.get("image"):
            content.append({"type": "image_url", "image_url": {"url": _data_url(_image_path(), "image/png")}})
        content.append({"type": "video_url", "video_url": {"url": _data_url(p["video"], "video/mp4")}})
        content.append({"type": "text", "text": p["text"]})
        body = {"model": MODEL_ID, "messages": [{"role": "user", "content": content}],
                "max_tokens": MAX_TOKENS, "temperature": 0.0,
                "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1200) as r:
            reply = json.load(r)
        msg = reply["choices"][0]["message"]
        text = msg.get("content") or ""
        if msg.get("reasoning_content"):
            text = msg["reasoning_content"] + "\n</think>\n\n" + text
        usage = reply.get("usage") or {}
        results.append({"prompt": i, "text": text, "prompt_tokens": usage.get("prompt_tokens")})
        print(f"[{args.tag}] prompt {i} ({usage.get('prompt_tokens')} prompt tokens): {text[:70]!r}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(results, indent=1))


def oracle(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=args.device, attn_implementation="sdpa").eval()
    tok = proc.tokenizer
    out = Path(args.out_dir)
    collected = {f.stem: json.loads(f.read_text()) for f in sorted(out.glob("*.json")) if f.stem != "report"}

    def inputs_for(p):
        content = []
        if p.get("image"):
            content.append({"type": "image", "image": Image.open(_image_path()).convert("RGB")})
        content.append({"type": "video", "video": p["video"]})
        content.append({"type": "text", "text": p["text"]})
        return proc.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

    def extend(inputs, ids):
        """Append continuation ids to every sequence-shaped input."""
        L = inputs["input_ids"].shape[1]
        ext = dict(inputs)
        add = torch.tensor([ids], device=model.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, L):
                if k == "input_ids":
                    fill = add
                else:
                    fill = torch.ones_like(add) if k == "attention_mask" else torch.zeros_like(add)
                ext[k] = torch.cat([v, fill.to(v.dtype)], dim=1)
        return ext, L

    report = {"oracle": {}, "oracle_prompt_tokens": {}, "served": {}}
    with torch.inference_mode():
        for i, p in enumerate(prompts(Path(args.data))):
            inputs = inputs_for(p)
            report["oracle_prompt_tokens"][i] = int(inputs["input_ids"].shape[1])
            gen = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
            ref_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
            report["oracle"][i] = tok.decode(ref_ids)
            print(f"oracle prompt {i} ({inputs['input_ids'].shape[1]} prompt tokens): {report['oracle'][i][:70]!r}")
            for tag, rows in collected.items():
                ids = tok(rows[i]["text"], add_special_tokens=False)["input_ids"][:MAX_TOKENS]
                if not ids:
                    continue
                ext, L = extend(inputs, ids)
                # only the forced positions: full logits of a long video prompt are GBs
                logits = model(**ext, logits_to_keep=len(ids) + 1).logits[0, :len(ids)].float()
                served = torch.tensor(ids, device=logits.device)
                top = logits.argmax(-1)
                agree = top == served
                first = int((~agree).nonzero()[0]) if (~agree).any() else None
                margin = None
                if first is not None:
                    margin = float(logits[first, top[first]] - logits[first, served[first]])
                n = min(len(ids), len(ref_ids))
                prefix = next((j for j in range(n) if ids[j] != ref_ids[j]), n)
                report["served"].setdefault(tag, {})[i] = {
                    "tokens": len(ids), "prompt_tokens": rows[i].get("prompt_tokens"),
                    "argmax_agreement": float(agree.float().mean()),
                    "first_disagreement": first, "logit_gap_at_first": margin,
                    "common_prefix_with_oracle_greedy": prefix}
                print(f"{tag} prompt {i}: agree={agree.float().mean():.3f} first={first} gap={margin} "
                      f"prefix={prefix}/{len(ref_ids)} prompt_tokens={rows[i].get('prompt_tokens')}")
    (out / "report.json").write_text(json.dumps(report, indent=1))

    print("\n| engine/run | argmax agreement (mean) | prompts fully argmax "
          "| first disagreement: logit gaps | prompt tokens vs HF |")
    print("|---|---:|---:|---|---|")
    for tag, rows in report["served"].items():
        vals = list(rows.values())
        mean = sum(v["argmax_agreement"] for v in vals) / len(vals)
        full = sum(v["first_disagreement"] is None for v in vals)
        gaps = ", ".join(f"{v['logit_gap_at_first']:.2f}" for v in vals if v["logit_gap_at_first"] is not None)
        diffs = [(v["prompt_tokens"] or 0) - report["oracle_prompt_tokens"][i] for i, v in rows.items()]
        print(f"| {tag} | {mean:.4f} | {full}/{len(vals)} | {gaps or '-'} | {diffs} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--port", type=int, required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--data", required=True)
    c.add_argument("--out-dir", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--data", required=True)
    o.add_argument("--out-dir", required=True)
    o.add_argument("--device", default="auto")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else oracle(a)
