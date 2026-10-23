#!/usr/bin/env python3
"""WER of several wav dirs in one ASR model load.

    python wer_dirs.py --sentences s.txt --device cuda:1 dir1 dir2 ...
Writes <dir>.wer.json next to each dir and prints one line per dir.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("WER_TREE", "/root/work/base"))
from benchmark import tts_speech_wer as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sentences", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("dirs", nargs="+")
args = ap.parse_args()

for d in args.dirs:
    out = str(Path(d).with_suffix(".wer.json"))
    W.main(["--audio-dir", d, "--sentences", args.sentences, "--device", args.device, "--out", out])
    r = json.load(open(out))
    keys = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
    print("WER", d, keys, flush=True)
