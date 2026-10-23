"""Softmax top-k routing in one kernel.

``softmax -> topk -> renormalize`` as separate ops is five launches per MoE layer
(``torch.topk`` alone is two, a radix select and a sort), which at decode batch sizes
costs more than the router GEMM. One program per token does it in registers.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_topk_kernel(logits, w_out, id_out, stride_t,
                         E: tl.constexpr, EP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                         NORM: tl.constexpr):
    """fp32 softmax over the E logits, the K largest probabilities in descending order,
    optionally divided by their sum. Equal probabilities (common: the logits are bf16) go
    to the lower expert first; ``torch.topk`` orders such ties arbitrarily, so the two can
    pick different experts at a tie, both equally valid."""
    t = tl.program_id(0)
    e = tl.arange(0, EP)
    valid = e < E
    x = tl.load(logits + t * stride_t + e, mask=valid, other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, 0)
    num = tl.where(valid, tl.exp(x), 0.0)
    prob = num / tl.sum(num, 0)
    cand = tl.where(valid, prob, -1.0)
    kk = tl.arange(0, KP)
    ids = tl.zeros([KP], tl.int32)
    ws = tl.zeros([KP], tl.float32)
    for j in tl.static_range(K):
        idx = tl.argmax(cand, 0)
        ws = tl.where(kk == j, tl.max(cand, 0), ws)
        ids = tl.where(kk == j, idx, ids)
        cand = tl.where(e == idx, -1.0, cand)
    if NORM:
        ws = ws / tl.sum(tl.where(kk < K, ws, 0.0), 0)
    tl.store(w_out + t * K + kk, ws, mask=kk < K)
    tl.store(id_out + t * K + kk, ids.to(tl.int64), mask=kk < K)


@torch.library.custom_op("mstar::moe_softmax_topk", mutates_args=())
def softmax_topk(logits: torch.Tensor, top_k: int, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k of ``softmax(logits (T, E))`` as ``(fp32 weights, int64 expert ids)``, the
    weights renormalized to sum to one when ``normalize``. Opaque to dynamo, so a compiled
    router keeps it whole."""
    T, E = logits.shape
    assert logits.stride(1) == 1
    topk_w = torch.empty(T, top_k, dtype=torch.float32, device=logits.device)
    topk_ids = torch.empty(T, top_k, dtype=torch.int64, device=logits.device)
    if T:
        _softmax_topk_kernel[(T,)](
            logits, topk_w, topk_ids, logits.stride(0), E=E, EP=triton.next_power_of_2(E),
            K=top_k, KP=triton.next_power_of_2(top_k), NORM=normalize, num_warps=1)
    return topk_w, topk_ids


@softmax_topk.register_fake
def _(logits, top_k, normalize):
    T = logits.shape[0]
    return (logits.new_empty(T, top_k, dtype=torch.float32),
            logits.new_empty(T, top_k, dtype=torch.int64))
