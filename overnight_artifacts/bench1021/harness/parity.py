"""Served greedy output, teacher-forced through a transformers oracle.

``collect`` asks a running server (mstar, vLLM or SGLang; OpenAI chat API) for
greedy completions of a fixed prompt set and saves them under a tag. Run it
against every engine, and against mstar twice in two server processes
(``mstar_a``, ``mstar_b``): the second is the same-code noise floor.

``oracle`` runs with no server on the GPUs. It loads the checkpoint in
transformers, generates its own greedy answer per prompt, then teacher-forces
every collected completion: at each position, is the served token the oracle's
argmax given the same prefix? Per completion it records

  argmax_agreement      share of served tokens that are the oracle's argmax (re-tokenization
                        artifacts, where the oracle's token spells the same text, count as agreeing
                        and are counted in retokenization_artifacts)
  first_disagreement    index of the first one that isn't (None = all agree)
  logit_gap_at_first    oracle logit(argmax) - logit(served) there: a near-tie is ~0.1-0.25 in bf16
  rank_at_first         rank of the served token in the oracle's distribution there;
                        vLLM's check_logprobs_close passes when it is < 5
  confident_agreement   agreement over positions where the oracle's top-2 margin > 1 (Qwen3-TTS
                        #302 gated on 100% here and only reported raw agreement): near-ties
                        can't flip these, so anything below 1.0 is a real divergence
  common_prefix         tokens shared with the oracle's own greedy answer
  prompt_tokens_delta   served usage.prompt_tokens - oracle prompt length (template / media
                        tokenization drift shows here first; it should be 0). None when the
                        server doesn't report usage, as mstar's chat API doesn't: then a
                        prompt-ids test against the HF processor has to cover it.

Greedy text is not byte-reproducible across engines or server processes, so
judge these against the control and the other engines (gate.py), not against 1.0.

prompts.jsonl, one object per line ("image"/"video" are local paths):
  {"text": "What is the capital of Australia?"}
  {"text": "Describe this image.", "image": "/data/cat.png"}
  {"text": "What happens in this clip?", "video": "/data/clip.mp4"}

  python parity.py collect --port 8300 --tag mstar_a --model-id Qwen/Qwen3.8-27B --prompts prompts.jsonl --out-dir out/parity
  python parity.py oracle --model-id Qwen/Qwen3.8-27B --prompts prompts.jsonl --out-dir out/parity
  python parity.py compare --model-id Qwen/Qwen3.8-27B --prompts prompts.jsonl --out-dir out/parity --a eager --b graphs
"""
import argparse
import base64
import json
import mimetypes
import urllib.request
from pathlib import Path


def load_prompts(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def collect(args) -> None:
    url = f"http://localhost:{args.port}/v1/chat/completions"
    chat_kwargs = json.loads(args.chat_kwargs)
    results = []
    for i, p in enumerate(load_prompts(args.prompts)):
        content = []
        if p.get("image"):
            content.append({"type": "image_url", "image_url": {"url": _data_url(p["image"])}})
        if p.get("video"):
            content.append({"type": "video_url", "video_url": {"url": _data_url(p["video"])}})
        content.append({"type": "text", "text": p["text"]})
        # mstar reads template kwargs at the top level; vLLM / SGLang under chat_template_kwargs
        body = {"model": args.model_id, "messages": [{"role": "user", "content": content}],
                "max_tokens": args.max_tokens, "temperature": 0.0, **chat_kwargs,
                "chat_template_kwargs": chat_kwargs}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            reply = json.load(r)
        msg = reply["choices"][0]["message"]
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:   # an engine with a reasoning parser split the turn; rejoin it
            text = reasoning + "\n</think>\n\n" + text
        usage = reply.get("usage") or {}
        results.append({"prompt": i, "text": text, "prompt_tokens": usage.get("prompt_tokens"),
                        "finish_reason": reply["choices"][0].get("finish_reason")})
        print(f"[{args.tag}] prompt {i} ({usage.get('prompt_tokens')} prompt tokens): {text[:70]!r}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(results, indent=1))


def oracle(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    # apply_torch_config-style TF32 would blur an fp32 yardstick
    torch.set_float32_matmul_precision("highest")
    dtype = getattr(torch, args.dtype)
    prompts = load_prompts(args.prompts)
    multimodal = any(p.get("image") or p.get("video") for p in prompts)
    if multimodal:
        proc = AutoProcessor.from_pretrained(args.model_id)
        tok = proc.tokenizer
        cls = AutoModelForImageTextToText
    else:
        proc = tok = AutoTokenizer.from_pretrained(args.model_id)
        cls = AutoModelForCausalLM
    # device_map= needs `accelerate`; .to() for one device keeps the env small
    kw = {"device_map": "auto"} if args.device == "auto" else {}
    model = cls.from_pretrained(args.model_id, dtype=dtype, attn_implementation=args.attn, **kw).eval()
    if args.device != "auto":
        model = model.to(args.device)
    chat_kwargs = json.loads(args.chat_kwargs)
    out = Path(args.out_dir)
    collected = {f.stem: json.loads(f.read_text()) for f in sorted(out.glob("*.json"))
                 if not f.stem.startswith("report")}
    if not collected:
        raise SystemExit(f"no collected answers in {out}")

    def inputs_for(p):
        content = []
        if p.get("image"):
            from PIL import Image   # only image prompts need PIL
            content.append({"type": "image", "image": Image.open(p["image"]).convert("RGB")})
        if p.get("video"):
            content.append({"type": "video", "video": p["video"]})
        content.append({"type": "text", "text": p["text"]})
        msgs = [{"role": "user", "content": content if multimodal else p["text"]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                        return_tensors="pt", **chat_kwargs).to(model.device)

    def extend(inputs, ids):
        """Append continuation ids to every sequence-shaped input."""
        L = inputs["input_ids"].shape[1]
        ext = dict(inputs)
        add = torch.tensor([ids], device=model.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.dim() == 2 and v.shape == (1, L):
                fill = add if k == "input_ids" else (torch.ones_like(add) if k == "attention_mask"
                                                     else torch.zeros_like(add))
                ext[k] = torch.cat([v, fill.to(v.dtype)], dim=1)
        return ext

    report = {"model_id": args.model_id, "dtype": args.dtype, "attn": args.attn,
              "oracle": {}, "oracle_prompt_tokens": {}, "served": {}}
    with torch.inference_mode():
        for i, p in enumerate(prompts):
            inputs = inputs_for(p)
            L = int(inputs["input_ids"].shape[1])
            report["oracle_prompt_tokens"][i] = L
            gen = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, num_beams=1)
            ref_ids = gen[0, L:].tolist()
            report["oracle"][i] = tok.decode(ref_ids)
            print(f"oracle prompt {i} ({L} prompt tokens): {report['oracle'][i][:70]!r}", flush=True)
            for tag, rows in collected.items():
                text = rows[i]["text"]
                ids = tok(text, add_special_tokens=False)["input_ids"][:args.max_tokens]
                if not ids:
                    report["served"].setdefault(tag, {})[i] = {"tokens": 0, "argmax_agreement": 0.0,
                                                                "first_disagreement": 0, "logit_gap_at_first": None,
                                                                "rank_at_first": None, "common_prefix": 0,
                                                                "prompt_tokens_delta": None, "empty": True}
                    continue
                # The served ids are re-tokenized from text, which can split it differently from
                # how it was sampled ("\n\n" as two "\n"; a word cut by max_tokens). A mis-split
                # is a fake disagreement and puts the oracle off-distribution after it. So where
                # the oracle's token spells the same text, adopt its split, re-tokenize the rest
                # and force again; only disagreements that change the text remain.
                artifacts = 0
                for _ in range(32):
                    # only the forced positions: full logits of a long multimodal prompt are GBs
                    logits = model(**extend(inputs, ids), logits_to_keep=len(ids) + 1).logits[0, :len(ids)].float()
                    served = torch.tensor(ids, device=logits.device)
                    top = logits.argmax(-1)
                    agree = top == served
                    realigned = False
                    for k in (~agree).nonzero().flatten().tolist():
                        alt_ids = ids[:k] + [int(top[k])]
                        alt = tok.decode(alt_ids)
                        if alt != tok.decode(ids[:k]) and (text.startswith(alt) or alt.startswith(text)):
                            rest = text[len(alt):]
                            ids = (alt_ids + (tok(rest, add_special_tokens=False)["input_ids"] if rest else []))
                            ids = ids[:args.max_tokens]
                            artifacts += 1
                            realigned = True
                            break
                    if not realigned:
                        break
                first = int((~agree).nonzero()[0]) if (~agree).any() else None
                # Confident positions (oracle top-2 margin > 1, #302's gate): bf16 near-ties
                # can't flip these, so any disagreement there is a real divergence.
                top2 = logits.topk(2, dim=-1).values
                confident = (top2[:, 0] - top2[:, 1]) > args.confident_margin
                n_conf = int(confident.sum())
                conf_agree = int(agree[confident].sum()) / n_conf if n_conf else None   # exact count ratio: a float32 mean gives 0.99999994 for all-agree
                gap = rank = None
                if first is not None:
                    row = logits[first]
                    gap = float(row[top[first]] - row[served[first]])
                    rank = int((row > row[served[first]]).sum())
                n = min(len(ids), len(ref_ids))
                prefix = next((j for j in range(n) if ids[j] != ref_ids[j]), n)
                # mstar's chat API reports usage as zeros (serving_chat.py): 0 means "not reported"
                pt = rows[i].get("prompt_tokens") or None
                report["served"].setdefault(tag, {})[i] = {
                    "tokens": len(ids), "argmax_agreement": float(agree.float().mean()),
                    "first_disagreement": first, "logit_gap_at_first": gap, "rank_at_first": rank,
                    "common_prefix": prefix, "prompt_tokens_delta": None if pt is None else pt - L,
                    "retokenization_artifacts": artifacts,
                    "confident_positions": n_conf, "confident_agreement": conf_agree}
                print(f"  {tag}: agree={agree.float().mean():.3f} confident={conf_agree}/{n_conf} first={first} "
                      f"gap={gap} rank={rank} artifacts={artifacts} "
                      f"prefix={prefix}/{len(ref_ids)} prompt_delta={None if pt is None else pt - L}", flush=True)
    name = "report.json" if args.dtype == "bfloat16" else f"report_{args.dtype}.json"
    (out / name).write_text(json.dumps(report, indent=1))

    print("\n| run | argmax agreement | confident agreement (min) | prompts fully argmax | gaps at first disagreement "
          "| ranks | prompt tokens vs oracle |")
    print("|---|---:|---:|---:|---|---|---|")
    for tag, rows in report["served"].items():
        vals = list(rows.values())
        mean = sum(v["argmax_agreement"] for v in vals) / len(vals)
        full = sum(v["first_disagreement"] is None for v in vals)
        gaps = ", ".join(f"{v['logit_gap_at_first']:.2f}" for v in vals if v["logit_gap_at_first"] is not None)
        ranks = ", ".join(str(v["rank_at_first"]) for v in vals if v["rank_at_first"] is not None)
        deltas = [v["prompt_tokens_delta"] for v in vals]
        conf = [v["confident_agreement"] for v in vals if v.get("confident_agreement") is not None]
        conf_s = f"{min(conf):.4f}" if conf else "-"
        print(f"| {tag} | {mean:.4f} | {conf_s} | {full}/{len(vals)} | {gaps or '-'} | {ranks or '-'} | {deltas} |")


def compare(args) -> None:
    """Two collected runs against each other, no oracle: setting equivalence (eager vs
    captured, TP1 vs TP2, alone vs under load). Run the same-code pair first; a
    difference is only evidence when it is outside what that pair shows."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    out = Path(args.out_dir)
    a, b = (json.loads((out / f"{t}.json").read_text()) for t in (args.a, args.b))
    same = 0
    for ra, rb in zip(a, b):
        ia, ib = (tok(r["text"], add_special_tokens=False)["input_ids"] for r in (ra, rb))
        n = min(len(ia), len(ib))
        first = next((j for j in range(n) if ia[j] != ib[j]), None if len(ia) == len(ib) else n)
        same += first is None
        where = "identical" if first is None else f"first differs at token {first} of {len(ia)}/{len(ib)}"
        print(f"prompt {ra['prompt']}: {where}")
    print(f"{args.a} vs {args.b}: {same}/{len(a)} identical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "oracle", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--model-id", required=True)
        s.add_argument("--prompts", required=True)
        s.add_argument("--out-dir", required=True)
        s.add_argument("--max-tokens", type=int, default=128)
        s.add_argument("--chat-kwargs", default='{"enable_thinking": false}',
                       help="chat-template kwargs, the same on every side (JSON)")
    c = sub.choices["collect"]
    c.add_argument("--port", type=int, required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--timeout", type=float, default=1200)
    o = sub.choices["oracle"]
    o.add_argument("--device", default="cuda:0", help="'auto' shards a model too big for one GPU")
    o.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    o.add_argument("--attn", default="sdpa")
    o.add_argument("--confident-margin", type=float, default=1.0,
                   help="oracle top-2 logit margin above which a position counts as confident")
    m = sub.choices["compare"]
    m.add_argument("--a", required=True, help="tag of the first collected run")
    m.add_argument("--b", required=True, help="tag of the second")
    a = ap.parse_args()
    {"collect": collect, "oracle": oracle, "compare": compare}[a.cmd](a)
