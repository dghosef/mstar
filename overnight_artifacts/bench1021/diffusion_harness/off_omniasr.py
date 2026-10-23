#!/usr/bin/env python3
"""Offline competitor for Omnilingual ASR CTC: the official omnilingual-asr 0.2.0 ASRInferencePipeline
(fairseq2 0.6, torch 2.8), bf16, v2 weights (the same files M* loads), argmax CTC, batch b =
pipeline.transcribe(paths, batch_size=b). Requests are length-sorted within a run (the strongest
offline arm: least padding); the pipeline decodes/resamples/normalizes the files itself, like a server.
Card: a local fairseq2 card on M*'s cached v2 files (--local-card) if it loads, else Meta's own card.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doff_common as D  # noqa: E402


def extra(ap):
    ap.add_argument("--card", required=True, help="e.g. omniASR_CTC_300M_v2")
    ap.add_argument("--local-card", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-sort", action="store_true", help="as shipped: batches in input order")


def main():
    a = D.parse(extra)
    import soundfile as sf
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    dt = getattr(torch, a.dtype)
    pipe = None
    if a.local_card:
        try:
            pipe = ASRInferencePipeline(model_card=a.local_card, device="cuda", dtype=dt)
            print(f"loaded local card {a.local_card}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"local card failed ({e!r}); using {a.card}", flush=True)
    if pipe is None:
        pipe = ASRInferencePipeline(model_card=a.card, device="cuda", dtype=dt)
    dur = {}

    def length(item):
        p = item["audio_path"]
        if p not in dur:
            dur[p] = sf.info(p).duration
        return dur[p]

    def order(reqs):
        return sorted(reqs, key=lambda r: -length(a.items_list[r[1]]))

    def process(batch):
        return list(pipe.transcribe([it["audio_path"] for _, it, _ in batch], batch_size=len(batch)))

    D.run(a, process, sync=torch.cuda.synchronize, order=None if a.no_sort else order)


if __name__ == "__main__":
    main()
