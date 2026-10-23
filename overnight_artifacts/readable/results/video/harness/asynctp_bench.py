"""Cosmos3-Super TP2 row-parallel shapes: matmul + NCCL all-reduce vs symm_mem fused
matmul+reduce-scatter (+ fused all-gather+matmul for the next column-parallel GEMM).

torchrun --nproc-per-node 2 asynctp_bench.py
"""
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

dist.init_process_group("nccl")
r, W = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
group = dist.group.WORLD
symm.enable_symm_mem_for_group(group.group_name)
M, H, F = 18720, 5120, 25600 // W  # tokens, hidden, local MLP width
x_full = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
w_up = torch.randn(H, 2 * F, device="cuda", dtype=torch.bfloat16) * 0.01   # gate+up, column-parallel
w_down = torch.randn(F, H, device="cuda", dtype=torch.bfloat16) * 0.01     # row-parallel


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def mlp_allreduce():
    h = x_full @ w_up
    a = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    y = a @ w_down
    dist.all_reduce(y)
    return y


x_shard = x_full[r * M // W:(r + 1) * M // W].contiguous()


def mlp_fused():
    # sequence-sharded residual in, sequence-sharded out (Megatron SP)
    _, (h,) = torch.ops.symm_mem.fused_all_gather_matmul(x_shard, [w_up], gather_dim=0, group_name=group.group_name)
    a = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    return torch.ops.symm_mem.fused_matmul_reduce_scatter(a, w_down, "sum", scatter_dim=0, group_name=group.group_name)


def mlp_nooverlap_rs_ag():
    xg = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
    dist.all_gather_into_tensor(xg, x_shard)
    h = xg @ w_up
    a = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    y = a @ w_down
    out = torch.empty(M // W, H, device="cuda", dtype=torch.bfloat16)
    dist.reduce_scatter_tensor(out, y)
    return out


def mlp_compute_only():
    h = x_full @ w_up
    a = torch.nn.functional.silu(h[:, :F]) * h[:, F:]
    return a @ w_down


for name, fn in [("compute only", mlp_compute_only), ("matmul+allreduce", mlp_allreduce),
                 ("AG+matmul, matmul+RS (nccl, no overlap)", mlp_nooverlap_rs_ag),
                 ("fused symm_mem AG-matmul / matmul-RS", mlp_fused)]:
    try:
        t = bench(fn)
        if r == 0:
            print(f"{name:42s} {t:7.3f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        if r == 0:
            print(f"{name}: {e!r}"[:300], flush=True)
ref = mlp_allreduce()[r * M // W:(r + 1) * M // W]
got = mlp_fused()
if r == 0:
    print("fused vs allreduce maxdiff", (ref.float() - got.float()).abs().max().item())
dist.destroy_process_group()
