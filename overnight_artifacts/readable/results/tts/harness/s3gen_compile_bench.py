"""S3Gen flow solve: CUDA graph of the eager fp16 estimator vs graph of a torch.compile'd one.

    CUDA_VISIBLE_DEVICES=1 python s3gen_compile_bench.py [--mode default|max-autotune-no-cudagraphs]
Times one 10-step CFG solve per (rows, frames) under each, and checks the outputs agree.
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "/root/work/head2")
sys.path.insert(0, "/root/work/head2/benchmark/chatterbox")
from profile_s3gen import capture, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="default")
a = ap.parse_args()

s3gen, ref, n_steps = load("standard", "cuda")
s3gen.decoder.set_estimator_dtype(torch.float16)
cfm = s3gen.decoder
n_mel = s3gen.config.output_size
results = {}


def inputs(rows, frames, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda *shape: torch.randn(*shape, device="cuda", generator=g)  # noqa: E731
    return dict(mu=mk(rows, n_mel, frames), mask=torch.ones(rows, 1, frames, device="cuda"),
                spks=mk(rows, n_mel), cond=mk(rows, n_mel, frames), noise=mk(rows, n_mel, frames))


def bench(label, shapes):
    for rows, frames in shapes:
        x = inputs(rows, frames)
        graph, out = capture(lambda: cfm.solve(**x, n_timesteps=n_steps))
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        results[(label, rows, frames)] = (s.elapsed_time(e) / 10, out.clone())
        print(label, rows, frames, f"{s.elapsed_time(e) / 10:.2f} ms", flush=True)


shapes = [(1, 448), (4, 448), (8, 448), (8, 704), (16, 448)]
with torch.inference_mode():
    bench("graph", shapes)
    t0 = time.time()
    cfm.estimator = torch.compile(cfm.estimator, dynamic=True, mode=a.mode)
    bench("compiled+graph", shapes)
    print(f"compile+capture wall {time.time() - t0:.0f} s")
for rows, frames in shapes:
    t1, o1 = results[("graph", rows, frames)]
    t2, o2 = results[("compiled+graph", rows, frames)]
    print(f"rows {rows} frames {frames}: {t1:.2f} -> {t2:.2f} ms ({100 * (t1 - t2) / t1:+.0f}%), "
          f"max|diff| {(o1 - o2).abs().max().item():.4f}, out max {o1.abs().max().item():.2f}")
