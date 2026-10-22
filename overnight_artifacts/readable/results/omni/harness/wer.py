#!/usr/bin/env python3
"""WER of Qwen3-Omni speech against its own text reply (runs on the box, on a free GPU).

  python wer.py DIR [DIR ...]   # each DIR holds req_N.txt + req_N.wav from benchmark.runner

Per DIR: whisper-large-v3 transcript of each wav (long-form, 30 s chunks), WER vs the
Thinker text in req_N.txt with Whisper's English normalizer on both sides. Writes
DIR/wer.json and prints a one-line summary. Also reports total audio seconds.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import jiwer
import soundfile as sf
import torch
from transformers import WhisperTokenizer, pipeline

ASR = "openai/whisper-large-v3"


def main() -> None:
    dirs = sys.argv[1:]
    asr = pipeline("automatic-speech-recognition", model=ASR, torch_dtype=torch.float16,
                   device=os.environ.get("WER_DEVICE", "cuda:0"))
    norm = WhisperTokenizer.from_pretrained(ASR).normalize
    for d in dirs:
        wavs = sorted(glob.glob(f"{d}/req_*.wav"))
        pairs = []
        for w in wavs:
            t = w[:-4] + ".txt"
            if os.path.exists(t):
                pairs.append((w, open(t).read()))
        if not pairs:
            print(f"{d}: no wav+txt pairs")
            continue
        hyps = asr([w for w, _ in pairs], chunk_length_s=30, batch_size=16,
                   generate_kwargs={"language": "en", "task": "transcribe"})
        rows, refs_n, hyps_n, secs = [], [], [], 0.0
        for (w, ref), h in zip(pairs, hyps):
            dur = sf.info(w).duration
            secs += dur
            r_n, h_n = norm(ref), norm(h["text"])
            if not r_n.strip():
                continue
            refs_n.append(r_n)
            hyps_n.append(h_n)
            rows.append({"wav": os.path.basename(w), "dur": dur, "wer": jiwer.wer(r_n, h_n),
                         "ref": ref, "hyp": h["text"]})
        wer = jiwer.wer(refs_n, hyps_n) if refs_n else float("nan")
        per = sorted(r["wer"] for r in rows)
        out = {"n": len(rows), "corpus_wer": wer, "median_wer": per[len(per) // 2] if per else None,
               "n_wer_gt_0.2": sum(x > 0.2 for x in per), "audio_s": secs, "rows": rows}
        json.dump(out, open(f"{d}/wer.json", "w"), indent=1)
        print(f"{d}: n={len(rows)} corpus_wer={wer:.4f} median={out['median_wer']} "
              f">0.2:{out['n_wer_gt_0.2']} audio_s={secs:.1f}", flush=True)


if __name__ == "__main__":
    main()
