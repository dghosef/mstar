"""Canary-1B-v2 speech translation gate (de -> en) on FLEURS: mstar vs NeMo 3.0.0 with its own prompt.

  prep   download FLEURS de_de test audio + tsv and en_us test tsv (same sentence ids), keep the first --n German
         test utterances that have an English reference, write <dir>/wav/*.wav and <dir>/manifest.json
  mstar  translate every clip through a running mstar Canary server (/v1/audio/transcriptions, language de,
         target_language en), write <dir>/mstar.json
  nemo   the same clips through NeMo's transcribe(source_lang="de", target_lang="en") (bf16 autocast), <dir>/nemo.json
  cmp    corpus BLEU (sacrebleu) of each against the English references, and how often mstar's text equals NeMo's.
         Pass (fixed before the run): mstar BLEU within 1.0 of NeMo's.

  python canary_translate_gate.py prep --dir D [--n 100]
  python canary_translate_gate.py mstar --dir D --url http://127.0.0.1:28300
  nemo/bin/python canary_translate_gate.py nemo --dir D
  nemo/bin/python canary_translate_gate.py cmp --dir D
"""
import argparse
import csv
import json
import tarfile
import urllib.request
import uuid
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prep", "mstar", "nemo", "cmp"])
ap.add_argument("--dir", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--url", default="http://127.0.0.1:28300")
a = ap.parse_args()
d = Path(a.dir)
d.mkdir(parents=True, exist_ok=True)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


if a.mode == "prep":
    from huggingface_hub import hf_hub_download

    get = lambda f: hf_hub_download("google/fleurs", f, repo_type="dataset")  # noqa: E731
    en = {r[0]: r[2] for r in rows(get("data/en_us/test.tsv"))}
    de = [(r[0], r[1], r[2]) for r in rows(get("data/de_de/test.tsv")) if r[0] in en]
    keep, seen = [], set()
    for sid, fname, text in de:
        if sid not in seen:
            seen.add(sid)
            keep.append((sid, fname, text))
        if len(keep) == a.n:
            break
    wanted = {fname for _, fname, _ in keep}
    (d / "wav").mkdir(exist_ok=True)
    with tarfile.open(get("data/de_de/audio/test.tar.gz")) as tar:
        for m in tar:
            name = Path(m.name).name
            if name in wanted:
                (d / "wav" / name).write_bytes(tar.extractfile(m).read())
    items = [{"id": sid, "wav": str(d / "wav" / fname), "de": text, "en": en[sid]} for sid, fname, text in keep
             if (d / "wav" / fname).exists()]
    (d / "manifest.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} clips")
elif a.mode == "mstar":
    items = json.loads((d / "manifest.json").read_text())
    out = {}
    for it in items:
        boundary = uuid.uuid4().hex
        fields = (("model", "canary"), ("language", "de"), ("target_language", "en"), ("temperature", "0"),
                  ("response_format", "json"))
        body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                        for k, v in fields)
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n").encode() + Path(it["wav"]).read_bytes()
        body += f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{a.url}/v1/audio/transcriptions", body,
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        out[it["id"]] = json.load(urllib.request.urlopen(req, timeout=300))["text"]
    (d / "mstar.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"mstar: {len(out)} translations")
elif a.mode == "nemo":
    import torch
    from nemo.collections.asr.models import ASRModel

    items = json.loads((d / "manifest.json").read_text())
    m = ASRModel.from_pretrained("nvidia/canary-1b-v2").cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hyps = m.transcribe([it["wav"] for it in items], batch_size=16, verbose=False, source_lang="de",
                            target_lang="en")
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    out = {it["id"]: (h.text if hasattr(h, "text") else str(h)) for it, h in zip(items, hyps, strict=True)}
    (d / "nemo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"nemo: {len(out)} translations")
else:
    import sacrebleu

    items = json.loads((d / "manifest.json").read_text())
    mstar, nemo = json.loads((d / "mstar.json").read_text()), json.loads((d / "nemo.json").read_text())
    ids = [it["id"] for it in items if it["id"] in mstar and it["id"] in nemo]
    refs = [[next(it["en"] for it in items if it["id"] == i) for i in ids]]
    b_m = sacrebleu.corpus_bleu([mstar[i] for i in ids], refs).score
    b_n = sacrebleu.corpus_bleu([nemo[i] for i in ids], refs).score
    b_mn = sacrebleu.corpus_bleu([mstar[i] for i in ids], [[nemo[i] for i in ids]]).score
    exact = sum(" ".join(mstar[i].split()) == " ".join(nemo[i].split()) for i in ids)
    verdict = "PASS" if abs(b_m - b_n) <= 1.0 else "FAIL"
    print(f"{len(ids)} clips: BLEU vs FLEURS en  mstar {b_m:.2f}  NeMo {b_n:.2f}  (diff {b_m - b_n:+.2f}) -> {verdict}")
    print(f"mstar vs NeMo: exact match {exact}/{len(ids)}, BLEU of mstar against NeMo's text {b_mn:.2f}")
    for i in [i for i in ids if " ".join(mstar[i].split()) != " ".join(nemo[i].split())][:5]:
        print(f"  {i}\n    mstar: {mstar[i][:160]}\n    nemo:  {nemo[i][:160]}")
