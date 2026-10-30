"""NeMo reference for a long-form file: Canary-1B-v2 (or another NeMo ASR model) transcribing one long WAV through
the model's own transcribe() (NeMo 3 chunks long audio itself for the multitask models), printing WER against the
reference with the same whisper normalizer the served runs use, and saving the hypothesis.

  CUDA_VISIBLE_DEVICES=4 nemo/bin/python nemo_longform.py --wav long.wav --refs references.tsv --out hyp.txt
"""
import argparse
import re
import time
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/canary-1b-v2")
ap.add_argument("--wav", required=True)
ap.add_argument("--refs", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = ASRModel.from_pretrained(a.model).cuda().eval()
kw = {"source_lang": "en", "target_lang": "en"} if "canary" in a.model else {}
t0 = time.perf_counter()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    hyp = m.transcribe([a.wav], batch_size=1, verbose=False, **kw)
hyp = hyp[0] if isinstance(hyp, tuple) else hyp
text = hyp[0].text if hasattr(hyp[0], "text") else str(hyp[0])
wall = time.perf_counter() - t0
Path(a.out).write_text(text)
ref = " ".join(line.split("\t", 1)[1] for line in Path(a.refs).read_text().splitlines() if line.strip())


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s.upper())).strip().split()


try:
    import jiwer

    wer = jiwer.wer(" ".join(norm(ref)), " ".join(norm(text)))
except ImportError:
    wer = float("nan")
print(f"{a.model}: {len(norm(text))} words vs {len(norm(ref))} reference, WER {wer * 100:.2f}% (basic normalizer), "
      f"wall {wall:.1f} s", flush=True)
