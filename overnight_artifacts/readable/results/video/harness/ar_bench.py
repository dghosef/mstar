"""2-GPU NCCL all-reduce of a Super-sized activation (18720 x 5120 bf16).

torchrun --nproc-per-node 2 ar_bench.py
"""
import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(r)
for rows in (18720, 9360, 4096):
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    if r == 0:
        mb = x.numel() * 2 / 1e6
        print(f"PROTO={os.environ.get('NCCL_PROTO', '-')} ALGO={os.environ.get('NCCL_ALGO', '-')} "
              f"rows={rows} {mb:.0f} MB: {dt * 1000:.3f} ms  algbw {mb / 1e3 / dt:.0f} GB/s", flush=True)
dist.destroy_process_group()
