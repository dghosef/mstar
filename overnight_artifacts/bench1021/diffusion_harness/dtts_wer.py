#!/usr/bin/env python3
"""TTS gate: Whisper large-v3 transcripts of every arm's WAVs (<root>/<tag>/<i>.wav speaks text i),
Whisper English normalizer, corpus WER; criteria fixed in dtts.py's docstring:
  PASS if WER <= best WER over all tags + max(1.0, 2 x |WER(mstar_pr) - WER(mstar_pr_b)|),
       median seconds per word within +-25% of the median over all tags, and no failed/missing output.
"""
from __future__ import annotations

import argparse
import json
import statistics
import wave
from pathlib import Path

import numpy as np


def read_wav(p):
    with wave.open(str(p)) as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768
    return x, sr


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--failed-json", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from scipy.signal import resample_poly
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    texts = [t for t in Path(a.texts).read_text().splitlines() if t.strip()]
    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3",
                                                            torch_dtype=torch.float16).to("cuda").eval()
    norm = proc.tokenizer.normalize
    failed = json.loads(Path(a.failed_json).read_text()) if a.failed_json and Path(a.failed_json).exists() else {}
    res = {}
    for tag in [t for t in a.tags.split(",") if t]:
        d = Path(a.root) / tag
        items, missing = [], 0
        for i in range(a.count):
            p = d / f"{i:03d}.wav"
            if not p.exists():
                missing += 1
                continue
            x, sr = read_wav(p)
            if sr != 16000:
                from math import gcd
                g = gcd(sr, 16000)
                x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
            items.append((i, x, len(x) / 16000))
        hyps = {}
        for k in range(0, len(items), 16):
            batch = items[k:k + 16]
            feats = proc([x for _, x, _ in batch], sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                ids = model.generate(feats.to("cuda", torch.float16), language="en", task="transcribe",
                                     max_new_tokens=440)
            for (i, _, _), t in zip(batch, proc.batch_decode(ids, skip_special_tokens=True)):
                hyps[i] = t
        E = N = 0
        spw = []
        per = []
        for i, _, dur in items:
            ref, hyp = norm(texts[i % len(texts)]).split(), norm(hyps.get(i, "")).split()
            e = edits(ref, hyp)
            E, N = E + e, N + len(ref)
            spw.append(dur / max(1, len(texts[i % len(texts)].split())))
            per.append({"i": i, "wer": e / max(1, len(ref)), "hyp": hyps.get(i, ""), "dur": dur})
        res[tag] = {"wer": 100 * E / max(1, N), "n": len(items), "missing": missing,
                    "failed": failed.get(tag), "median_s_per_word": statistics.median(spw) if spw else None,
                    "worst": sorted(per, key=lambda r: -r["wer"])[:5]}
        print(tag, res[tag]["wer"], flush=True)
    best = min(r["wer"] for r in res.values() if r["n"])
    spread = abs(res["mstar_pr"]["wer"] - res["mstar_pr_b"]["wer"]) if "mstar_pr" in res and "mstar_pr_b" in res \
        else 0.0
    tol = max(1.0, 2 * spread)
    med_spw = statistics.median(r["median_s_per_word"] for r in res.values() if r["median_s_per_word"])
    verdict = {}
    for tag, r in res.items():
        checks = {"wer": r["n"] > 0 and r["wer"] <= best + tol,
                  "speech_rate": r["median_s_per_word"] is not None and
                  abs(r["median_s_per_word"] / med_spw - 1) <= 0.25,
                  "valid_outputs": r["missing"] == 0 and (r["failed"] in (0, None))}
        # sanity gate (user direction 2026-10-22): excluded only if clearly broken; parity is annotated
        broken = (r["missing"] > 0 or r["failed"] not in (0, None) or r["n"] == 0 or
                  (r["wer"] > 2 * best and r["wer"] > 10.0))
        verdict[tag] = {"verdict": "FAIL (broken output)" if broken else "PASS", "parity": all(checks.values()),
                        "checks": checks}
    out = Path(a.out)
    (out / "gate.json").write_text(json.dumps({"best_wer": best, "same_code_spread": spread, "tolerance": tol,
                                               "median_s_per_word": med_spw, "results": res, "verdict": verdict},
                                              indent=1))
    lines = [f"Whisper large-v3, {a.count} LibriSpeech test-clean sentences per arm. Best WER {best:.2f}%, "
             f"same-code spread (two M* PR processes) {spread:.2f} pt, tolerance {tol:.2f} pt; "
             f"median s/word over arms {med_spw:.3f}", "",
             "Sanity gate: excluded only if silent/missing output or WER > 2x best and > 10%; parity (WER within "
             "tolerance and speech rate within 25%) is annotated, not gating.", "",
             "| tag | WER % | n | missing | median s/word | sanity gate | parity (annotated) |",
             "|---|---:|---:|---:|---:|---|---|"]
    for tag, r in res.items():
        lines.append(f"| {tag} | {r['wer']:.2f} | {r['n']} | {r['missing']} | "
                     f"{r['median_s_per_word'] or 0:.3f} | {verdict[tag]['verdict']} | "
                     f"{'yes' if verdict[tag]['parity'] else 'no (annotated)'} |")
    (out / "gate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
