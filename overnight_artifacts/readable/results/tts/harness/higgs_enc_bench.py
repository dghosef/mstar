"""Higgs audio tower on short (< 4 s) chunks: eager vs torch.compile(dynamic=True).

    CUDA_VISIBLE_DEVICES=1 python higgs_enc_bench.py
"""
import sys
import time

import torch

sys.path.insert(0, "/root/work/head2")
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel  # noqa: E402

m = HiggsAudioModel("bosonai/higgs-audio-v3-stt")
enc = m.get_submodule("audio_encoder", device="cuda")
dtype = next(enc.audio_tower.parameters()).dtype


def timeit(fn, x, n=20):
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


with torch.inference_mode():
    xs = {w: torch.randn(1, 128, w, device="cuda", dtype=dtype) for w in (150, 237, 300, 389)}
    for w, x in xs.items():
        print("eager", w, f"{timeit(enc._encode, x):.2f} ms")
    comp = torch.compile(enc._encode, dynamic=True)
    t0 = time.time()
    comp(xs[300])
    comp(xs[150])
    print(f"compile {time.time() - t0:.0f} s")
    for w, x in xs.items():
        t1 = time.time()
        ms = timeit(comp, x)
        diff = (comp(x).float() - enc._encode(x).float()).abs().max().item()
        print("compiled", w, f"{ms:.2f} ms", f"first-call+bench {time.time() - t1:.1f} s", f"max|diff| {diff:.4f}")
