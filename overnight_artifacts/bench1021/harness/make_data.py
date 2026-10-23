"""Build the text workloads once per tokenizer; every engine reads the same files.

sharegpt.jsonl   ShareGPT V3 (anon8231489123/ShareGPT_Vicuna_unfiltered,
                 ShareGPT_V3_unfiltered_cleaned_split.json), sampled the way
                 `vllm bench serve --dataset-name sharegpt` does: first human turn as the
                 prompt, the first assistant reply's token length as the output length
                 (forced with ignore_eos), drop prompt<4 or output<4 tokens, drop
                 prompt>1024 or prompt+output>2048; shuffled with seed 0.
random_1024_256.jsonl   1024 random-vocabulary tokens decoded to text (re-encoded to
                 exactly 1024 where possible), 256 output tokens; seed 0.

  python make_data.py --tokenizer Qwen/Qwen3.5-0.8B --sharegpt <json> --out data/
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def sharegpt(tok, path, n):
    data = json.loads(Path(path).read_text())
    data = [d for d in data if len(d.get("conversations", [])) >= 2]
    random.Random(0).shuffle(data)
    rows = []
    for d in data:
        conv = d["conversations"]
        if conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
            continue
        p, c = conv[0]["value"], conv[1]["value"]
        pl = len(tok(p, add_special_tokens=False)["input_ids"])
        ol = len(tok(c, add_special_tokens=False)["input_ids"])
        if pl < 4 or ol < 4 or pl > 1024 or pl + ol > 2048:
            continue
        rows.append({"id": f"sg{len(rows)}", "prompt": p, "output_len": ol, "input_len": pl})
        if len(rows) >= n:
            break
    return rows


def random_rows(tok, n, in_len, out_len):
    rng = np.random.default_rng(0)
    vocab = tok.vocab_size  # base vocabulary: no added/special tokens
    special = set(tok.all_special_ids)
    rows = []
    for i in range(n):
        ids = [int(x) for x in rng.integers(0, vocab, in_len) if int(x) not in special]
        text = tok.decode(ids)
        for _ in range(10):
            re = tok(text, add_special_tokens=False)["input_ids"]
            if len(re) == in_len:
                break
            if len(re) > in_len:
                re = re[:in_len]
            else:
                re = re + [int(x) for x in rng.integers(0, vocab, in_len - len(re))]
            text = tok.decode(re)
        pl = len(tok(text, add_special_tokens=False)["input_ids"])
        rows.append({"id": f"rnd{i}", "prompt": text, "output_len": out_len, "input_len": pl})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workloads", default="sharegpt,random_1024_256",
                    help="sharegpt and/or random_<in>_<out> names; existing files are kept")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def build(name):
        if name == "sharegpt":
            return sharegpt(tok, a.sharegpt, a.n)
        _, i, o = name.split("_")
        return random_rows(tok, a.n, int(i), int(o))

    for name in a.workloads.split(","):
        if (out / f"{name}.jsonl").exists():
            continue
        rows = build(name)
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        il = [r["input_len"] for r in rows]
        ol = [r["output_len"] for r in rows]
        print(f"{name}: {len(rows)} rows; input mean {np.mean(il):.0f} (min {min(il)} max {max(il)}); "
              f"output mean {np.mean(ol):.0f} (min {min(ol)} max {max(ol)})")


if __name__ == "__main__":
    main()
