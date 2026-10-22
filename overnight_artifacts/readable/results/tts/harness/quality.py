#!/usr/bin/env python3
"""Correctness check of a TTS arm against a base arm.

    python quality.py --sentences s156.txt --base DIR_A --test DIR_B [--out q.json] [--device cuda:1]

WER of each arm (whisper-large-v3-turbo via benchmark.tts_speech_wer, every file = one
sample, so a sentence file with repeats gives several samples per prompt), and for files
present in both dirs the waveform agreement: duration ratio and normalized correlation
over the common length (1.0 = identical up to scale).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/root/work/mstar")


def wer_of(audio_dir: Path, sentences: dict, device: str, cache: dict):
    from benchmark.tts_speech_wer import transcribe, word_errors
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    wavs = sorted(audio_dir.glob("*.wav"))
    texts = transcribe(wavs, "openai/whisper-large-v3-turbo", device, 16)
    if "norm" not in cache:
        from transformers import WhisperProcessor
        proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        cache["norm"] = EnglishTextNormalizer(proc.tokenizer.english_spelling_normalizer)
    norm = cache["norm"]
    errs = words = 0
    per = {}
    for wav, hyp in zip(wavs, texts, strict=True):
        ref = norm(sentences[int(wav.stem)]).split()
        h = norm(hyp).split()
        e = word_errors(ref, h)
        errs += e
        words += len(ref)
        per[wav.stem] = {"wer": e / max(1, len(ref)), "hyp": hyp}
    return errs / max(1, words), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = Path(a.sentences).read_text().splitlines()
    sentences = {i + 1: ln.strip() for i, ln in enumerate(lines) if ln.strip()}
    cache = {}
    res = {}
    for name, d in (("base", a.base), ("test", a.test)):
        w, per = wer_of(Path(d), sentences, a.device, cache)
        res[name] = {"wer": w, "files": len(per), "per": per}
    corr, ratio = [], []
    for f in sorted(Path(a.base).glob("*.wav")):
        g = Path(a.test) / f.name
        if not g.exists():
            continue
        x, _ = sf.read(f, dtype="float32")
        y, _ = sf.read(g, dtype="float32")
        n = min(len(x), len(y))
        ratio.append(len(y) / max(1, len(x)))
        if n == 0:
            continue
        xs, ys = x[:n], y[:n]
        den = float(np.linalg.norm(xs) * np.linalg.norm(ys)) or 1.0
        corr.append(float(np.dot(xs, ys) / den))
    res["wave"] = {
        "pairs": len(corr),
        "corr_median": float(np.median(corr)) if corr else None,
        "corr_min": float(np.min(corr)) if corr else None,
        "identical_frac": float(np.mean([c > 0.9999 for c in corr])) if corr else None,
        "dur_ratio_median": float(np.median(ratio)) if ratio else None,
    }
    summary = {"base_wer": res["base"]["wer"], "test_wer": res["test"]["wer"], **res["wave"]}
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
