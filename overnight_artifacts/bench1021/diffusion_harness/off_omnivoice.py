#!/usr/bin/env python3
"""Offline competitor for OmniVoice: the official omnivoice package (git 08be0b4, the build M* imports its
FlashInfer forward from), fp16, apply_flashinfer (README: fastest at batch >= 4; --cuda-graph = its CUDA-graph
option), auto voice (no ref / instruct, like M*'s request), num_step 32, guidance 2.0, silence trim on (its
default, as M*'s). Batch b = texts per generate() call; TTFA = batch latency (one canvas, no streaming).
"""
import io
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doff_common as D  # noqa: E402


def extra(ap):
    ap.add_argument("--cuda-graph", action="store_true")


def main():
    a = D.parse(extra)
    for p in filter(None, os.environ.get("DGEN_OVERLAY", "").split(":")):
        sys.path.insert(0, p)
    import numpy as np
    import torch
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    try:
        apply_flashinfer(m, enable_cuda_graph=a.cuda_graph)
    except TypeError:
        apply_flashinfer(m)
    sr = int(getattr(m, "sampling_rate", 24000) or 24000)

    def wav(x):
        x = np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1, 1)
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((x * 32767).astype("<i2").tobytes())
        return b.getvalue(), len(x) / sr

    def process(batch):
        torch.manual_seed(batch[0][2])
        outs = m.generate(text=[it["text"] for _, it, _ in batch], num_step=32, guidance_scale=2.0)
        return [wav(o.cpu().numpy() if hasattr(o, "cpu") else o) for o in outs]

    D.run(a, process, sync=torch.cuda.synchronize, ext="wav")


if __name__ == "__main__":
    main()
