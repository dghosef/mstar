"""Time Wan2.2's mp4 mux (PyAV libx264) on a real decoded clip, with encoder thread variants.

python mux_bench.py <some.mp4 to take frames from>
"""
import io
import sys
import time
from fractions import Fraction

import av
import numpy as np

src = sys.argv[1]
with av.open(src) as c:
    frames = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
print("frames", frames.shape)


def mux(frames, thread_type=None, threads=None, options=None):
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=Fraction(24))
    stream.height, stream.width = frames.shape[1], frames.shape[2]
    stream.pix_fmt = "yuv420p"
    if thread_type:
        stream.codec_context.thread_type = thread_type
    if threads is not None:
        stream.codec_context.thread_count = threads
    if options:
        stream.codec_context.options = options
    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def decode(b):
    with av.open(io.BytesIO(b)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse) if mse else float("inf")


variants = [("default", {}), ("AUTO", {"thread_type": "AUTO"}), ("AUTO t8", {"thread_type": "AUTO", "threads": 8}),
            ("FRAME t16", {"thread_type": "FRAME", "threads": 16}), ("SLICE t8", {"thread_type": "SLICE", "threads": 8})]
s = mux(frames)
st = None
for name, kw in variants:
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = mux(frames, **kw)
        ts.append(time.perf_counter() - t0)
    q = psnr(frames, decode(out))
    print(f"{name:10s} {min(ts) * 1000:7.0f} ms  bytes {len(out)}  psnr_vs_src {q:.2f}", flush=True)

# the model's own postprocess, from the [1, 3, F, H, W] uint8 tensor the decoder emits
import torch  # noqa: E402

from mstar.model.wan22.wan22_model import Wan22Model  # noqa: E402

model = Wan22Model(model_path_hf="unused", skip_weight_loading=True)
t = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
for _ in range(2):
    t0 = time.perf_counter()
    model.postprocess(t, "video", {})
    print(f"model.postprocess {1000 * (time.perf_counter() - t0):7.0f} ms (torch threads {torch.get_num_threads()})")
t0 = time.perf_counter()
x = t[0].permute(1, 2, 3, 0).cpu().numpy()
y = np.ascontiguousarray(x)
print(f"permute+contiguous {1000 * (time.perf_counter() - t0):7.0f} ms")
for nt in (96, 8, 1):
    torch.set_num_threads(nt)
    t0 = time.perf_counter()
    z = t[0].permute(1, 2, 3, 0).contiguous().numpy()
    print(f"torch contiguous threads={nt} {1000 * (time.perf_counter() - t0):7.0f} ms")
t0 = time.perf_counter()
mux(z, thread_type="AUTO")
print(f"contiguous + AUTO mux {1000 * (time.perf_counter() - t0):7.0f} ms")
