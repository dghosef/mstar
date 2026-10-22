"""Model.load_video frames with 1 vs 8 FFmpeg threads: identical? how fast? (single decode, as the one preprocess thread runs it)"""
import sys
import time

import torch
from torchcodec.decoders import VideoDecoder

path = sys.argv[1]
res = {}
for n in (1, 8, 1, 8):
    t0 = time.perf_counter()
    frames = VideoDecoder(path, device="cpu", num_ffmpeg_threads=n).get_frames_in_range(0, 10**9).data
    dt = time.perf_counter() - t0
    res.setdefault(n, []).append(dt)
    if n == 1:
        ref = frames
    else:
        print(f"threads={n} identical={torch.equal(frames, ref)} shape={tuple(frames.shape)}")
print({k: [round(x, 3) for x in v] for k, v in res.items()})
