#!/usr/bin/env python3
"""CUDA-graph replay time of the Qwen3-TTS codec decode per (batch, window).

    PYTHONPATH=/root/work/base python codec_micro.py --model qwen3_tts [--dtype bfloat16]
"""
import argparse
import json

import torch

from mstar.model.registry import get_model_class, model_init_kwargs

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3_tts")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--bs", default="1,2,3,4,5,6,8,12,16,24,32")
ap.add_argument("--out", default=None)
ap.add_argument("--cudnn-bench", action="store_true")
ap.add_argument("--widths", default=None)
args = ap.parse_args()

kw = model_init_kwargs(args.model)
kw["codec_dtype"] = args.dtype
model = get_model_class(args.model)(**kw)
sub = model.get_submodule("Codec", device="cuda:0")
print("windows", sub.windows, "conv_context", sub.conv_context)
torch.backends.cudnn.benchmark = args.cudnn_bench
nq = model.config.codec.num_quantizers
res = {}
with torch.inference_mode():
    for width in ([int(w) for w in args.widths.split(',')] if args.widths else sub.windows):
        for bs in [int(x) for x in args.bs.split(",")]:
            codes = torch.randint(0, 2000, (bs, nq, width), device="cuda:0")
            conv_start = torch.zeros(bs, dtype=torch.long, device="cuda:0")
            for _ in range(2):
                sub._decode(codes, conv_start)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sub._decode(codes, conv_start)
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            n = 20
            s.record()
            for _ in range(n):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e) / n
            res[f"{width}x{bs}"] = ms
            print(f"width={width:3d} conv={sub.conv_frames(width):3d} bs={bs:3d} {ms:7.2f} ms  {ms/bs:6.2f} ms/row", flush=True)
            del g
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
