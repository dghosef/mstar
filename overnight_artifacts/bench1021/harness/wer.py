"""WER for the Qwen3-Omni workloads (Whisper English normalizer on both sides, jiwer).

  a2t   transcripts saved by client.py (--save-text) against the LibriSpeech references
        in omni_a2t.jsonl:
          python -I wer.py a2t --results <cell>/results.json --rows omni_a2t.jsonl --out wer.json
  t2s   speech written by benchmark.runner (req_<id>.wav + req_<id>.txt per request) is
        transcribed with openai/whisper-large-v3-turbo and compared with the text the same
        engine streamed (its own transcript), so a broken vocoder or talker shows up as WER;
        also counts silent / empty WAVs:
          python -I wer.py t2s --dir <runner output dir> --out wer.json
"""
import argparse
import json
import sys
from pathlib import Path


def normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    return EnglishTextNormalizer({})


def wer(refs, hyps):
    import jiwer
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return None
    return jiwer.wer([p[0] for p in pairs], [p[1] if p[1].strip() else "<empty>" for p in pairs])


def a2t(a):
    norm = normalizer()
    rows = {r["id"]: r for r in (json.loads(x) for x in Path(a.rows).read_text().splitlines() if x.strip())}
    recs = json.loads(Path(a.results).read_text())["requests"]
    refs, hyps = [], []
    for r in recs:
        if r["id"] in rows and r.get("ok"):
            refs.append(norm(rows[r["id"]]["reference"]))
            hyps.append(norm(r.get("text", "")))
    w = wer(refs, hyps)
    out = {"kind": "a2t", "n": len(refs), "wer": w}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"A2T WER n={len(refs)} wer={w}")


def t2s(a):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import pipeline
    norm = normalizer()
    d = Path(a.dir)
    wavs = sorted(d.glob("req_*.wav"))
    asr = pipeline("automatic-speech-recognition", model=a.asr_model, torch_dtype=torch.float16, device="cuda:0")
    refs, hyps, silent, durs = [], [], 0, []
    for w in wavs:
        txt = w.with_suffix(".txt")
        audio, sr = sf.read(w, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        durs.append(len(audio) / sr)
        if len(audio) == 0 or float(np.abs(audio).max()) < 1e-3:
            silent += 1
        hyp = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs={"language": "en"},
                  return_timestamps=len(audio) / sr > 30)["text"]
        refs.append(norm(txt.read_text() if txt.exists() else ""))
        hyps.append(norm(hyp))
    w = wer(refs, hyps)
    out = {"kind": "t2s", "n": len(wavs), "wer": w, "silent": silent,
           "mean_audio_s": float(np.mean(durs)) if durs else None, "asr_model": a.asr_model}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"T2S WER n={len(wavs)} wer={w} silent={silent}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("a2t")
    s1.add_argument("--results", required=True)
    s1.add_argument("--rows", required=True)
    s1.add_argument("--out", required=True)
    s2 = sub.add_parser("t2s")
    s2.add_argument("--dir", required=True)
    s2.add_argument("--out", required=True)
    s2.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    a = ap.parse_args()
    {"a2t": a2t, "t2s": t2s}[a.cmd](a)
    sys.exit(0)
