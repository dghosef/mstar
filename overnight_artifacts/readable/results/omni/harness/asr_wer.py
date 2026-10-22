#!/usr/bin/env python3
"""ASR WER of the Thinker's text on LibriSpeech requests (a2t/a2s dirs from benchmark.runner).

  python asr_wer.py DIR [...]   req_N.txt vs openslr/librispeech_asr clean validation row N
"""
import glob
import json
import os
import re
import sys

import jiwer
from datasets import load_dataset
from transformers import WhisperTokenizer

norm = WhisperTokenizer.from_pretrained("openai/whisper-large-v3").normalize
raw = load_dataset("openslr/librispeech_asr", "clean", split="validation",
                   cache_dir="/root/work/bench-cache", trust_remote_code=True)
refs = [r["text"] for r in raw.select(range(100))]
for d in sys.argv[1:]:
    hyps, gts = [], []
    for f in sorted(glob.glob(f"{d}/req_*.txt")):
        n = int(re.search(r"req_(\d+)\.txt", f).group(1))
        hyps.append(norm(open(f).read()))
        gts.append(norm(refs[n % len(refs)]))
    if not hyps:
        continue
    w = jiwer.wer(gts, hyps)
    per = sorted(((jiwer.wer(g, h), f) for g, h, f in zip(gts, hyps, sorted(glob.glob(f"{d}/req_*.txt")))), reverse=True)
    print("  worst:", [(round(a, 2), os.path.basename(b)) for a, b in per[:3]])
    # a clip phrased as a question ("how would the papers talk about it") gets an
    # answer rather than a transcript; count those apart instead of letting one
    # dominate the corpus number
    keep = [(g, h) for g, h in zip(gts, hyps) if jiwer.wer(g, h) <= 1.0]
    w_keep = jiwer.wer([g for g, _ in keep], [h for _, h in keep]) if keep else float("nan")
    json.dump({"n": len(hyps), "asr_wer": w, "asr_wer_transcribed": w_keep,
               "answered_instead": len(hyps) - len(keep)}, open(f"{d}/asr_wer.json", "w"))
    print(f"{d}: n={len(hyps)} ASR WER={w:.4f} transcribed-only={w_keep:.4f} "
          f"answered-instead={len(hyps) - len(keep)}", flush=True)
