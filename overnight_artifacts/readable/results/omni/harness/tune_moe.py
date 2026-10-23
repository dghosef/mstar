#!/usr/bin/env python3
"""Sweep fused_moe_kernel tiles for Qwen3-Omni's two MoE shapes (runs on the box, one GPU).

For each (shape, M) times GEMM1 (gate_up) and GEMM2 (down) separately per config with
CUDA-graph timing, and prints the default config's time next to the best.
Output: JSON {shape: {M: {"gemm1": cfg, "gemm2": cfg, ...}}}.
"""
import itertools
import json
import sys

import torch
import triton
from triton.testing import do_bench_cudagraph

from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import get_default_config, invoke_fused_moe_kernel
from mstar.utils.fused_moe.runner import _tl_compute_type

SHAPES = {"thinker": dict(E=128, H=2048, I=768, K=8), "talker": dict(E=128, H=1024, I=384, K=6)}
MS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "1,2,4,8,16,32,64".split(","))]


def configs(m):
    bms = [16] if m <= 16 else [16, 32, 64]
    for bm, bn, bk, w, s in itertools.product(bms, [32, 64, 128, 256], [64, 128, 256], [4, 8], [2, 3, 4, 5]):
        if bn * bk > 256 * 128 and s > 3:
            continue
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 1,
               "num_warps": w, "num_stages": s}


def main():
    out = {}
    torch.manual_seed(0)
    dev = "cuda"
    only = sys.argv[3] if len(sys.argv) > 3 else None
    for name, sh in SHAPES.items():
        if only and name != only:
            continue
        E, H, I, K = sh["E"], sh["H"], sh["I"], sh["K"]
        w1 = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
        w2 = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.02
        out[name] = {}
        for m in MS:
            x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
            ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(m)]).to(torch.int32)
            tw = torch.softmax(torch.randn(m, K, device=dev), -1)
            c1 = torch.empty(m * K, 2 * I, device=dev, dtype=torch.bfloat16)
            c2 = torch.randn(m * K, I, device=dev, dtype=torch.bfloat16)
            c3 = torch.empty(m * K, H, device=dev, dtype=torch.bfloat16)
            ct = _tl_compute_type(torch.bfloat16)
            res = {}
            per_bm = {}
            for gemm in ("gemm1", "gemm2"):
                best = None
                default = dict(get_default_config(m, E, 2 * I, H, K))
                for cfg in [default, *configs(m)]:
                    align = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], E)

                    def run(cfg=cfg, align=align):
                        if gemm == "gemm1":
                            invoke_fused_moe_kernel(A=x, B=w1, C=c1, topk_weights=tw, topk_ids=ids,
                                                    sorted_token_ids=align[0], expert_ids=align[1],
                                                    num_tokens_post_padded=align[2], mul_routed_weight=False,
                                                    top_k=K, config=cfg, compute_type=ct)
                        else:
                            invoke_fused_moe_kernel(A=c2, B=w2, C=c3, topk_weights=tw, topk_ids=ids,
                                                    sorted_token_ids=align[0], expert_ids=align[1],
                                                    num_tokens_post_padded=align[2], mul_routed_weight=True,
                                                    top_k=1, config=cfg, compute_type=ct)
                    try:
                        run()
                        t = do_bench_cudagraph(run, rep=30)
                    except Exception:
                        continue
                    if cfg is default:
                        res[f"{gemm}_default_us"] = t * 1e3
                    if best is None or t < best[0]:
                        best = (t, cfg)
                    if cfg is not default:
                        cur = per_bm.setdefault(cfg["BLOCK_SIZE_M"], {})
                        if gemm not in cur or t < cur[gemm][0]:
                            cur[gemm] = (t, cfg)
                res[f"{gemm}_best_us"] = best[0] * 1e3
            # both GEMMs share one alignment, so one BLOCK_SIZE_M
            bm = min((v["gemm1"][0] + v["gemm2"][0], k) for k, v in per_bm.items() if len(v) == 2)[1]
            res["gemm1"], res["gemm2"] = per_bm[bm]["gemm1"][1], per_bm[bm]["gemm2"][1]
            res["joint_us"] = (per_bm[bm]["gemm1"][0] + per_bm[bm]["gemm2"][0]) * 1e3
            out[name][m] = res
            print(name, m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
