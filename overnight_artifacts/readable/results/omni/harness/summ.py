#!/usr/bin/env python3
"""Summarize bench.py results: python summ.py /workspace/results/<tag> [...]"""
import glob
import json
import os
import re
import sys


def parse(log: str) -> dict:
    i = log.find("--- Benchmark Results")
    blk = log[i:] if i >= 0 else ""
    out = {}
    m = re.search(r"wall time: ([\d.]+)s", blk)
    out["wall"] = float(m.group(1)) if m else None
    for mod in ("text", "audio"):
        m = re.search(rf"TTFT \({mod}\)\s*: (.*)", blk)
        if m:
            mm = re.search(r"mean=([\d.]+)", m.group(1))
            p50 = re.search(r"p50=([\d.]+)", m.group(1))
            out[f"ttft_{mod}"] = float(mm.group(1)) if mm else None
            out[f"ttft_{mod}_p50"] = float(p50.group(1)) if p50 else None
    m = re.search(r"([\d.]+) text tok/s", blk)
    out["tok_s"] = float(m.group(1)) if m else None
    m = re.search(r"([\d.]+) audio sec/s", blk)
    out["audio_s_s"] = float(m.group(1)) if m else None
    m = re.search(r"([\d.]+) req/s", blk)
    out["req_s"] = float(m.group(1)) if m else None
    m = re.search(r"RTF\s*: mean=([\d.na/]+)\s+p50=([\d.na/]+)", blk)
    out["rtf_p50"] = m.group(2) if m else None
    m = re.search(r"Audio dur\s*: mean=([\d.]+)", blk)
    out["audio_dur"] = float(m.group(1)) if m else None
    m = re.search(r"Text tokens: (\d+) total \(([\d.]+) avg", blk)
    out["tok_avg"] = float(m.group(2)) if m else None
    return out


def main():
    for tag in sys.argv[1:]:
        for wl_dir in sorted(glob.glob(f"{tag}/*/*/bench.log")):
            d = os.path.dirname(wl_dir)
            arm = os.path.basename(os.path.dirname(d))
            wl = os.path.basename(d)
            r = parse(open(wl_dir).read())
            failed = None
            try:
                j = json.load(open(f"{d}/results.json"))
                failed = j.get("failed", j.get("n_failed"))
            except Exception:
                pass
            w = ""
            if os.path.exists(f"{d}/wer.json"):
                wj = json.load(open(f"{d}/wer.json"))
                w = f" wer={wj['corpus_wer']:.3f}"
            print(f"{os.path.basename(tag):10s} {arm:12s} {wl:9s} failed={failed} "
                  f"ttft_t={r.get('ttft_text')} ttft_a={r.get('ttft_audio')} tok/s={r['tok_s']} "
                  f"aud/s={r['audio_s_s']} req/s={r['req_s']} rtf50={r['rtf_p50']} "
                  f"dur={r['audio_dur']} tok_avg={r['tok_avg']}{w}")


if __name__ == "__main__":
    main()
