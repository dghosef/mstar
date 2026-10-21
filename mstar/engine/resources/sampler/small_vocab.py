"""Temperature, top-k, top-p and the draw in one kernel, for small vocabularies.

The graphable sampler's general path is a softmax kernel, a ``top_k`` fix-up,
FlashInfer's radix top-k renorm and its top-p rejection sampler, and a cast:
five launches and ~25 us per call at decode batch sizes, which dominates for a
codec head with a 2-4k vocabulary called once per codebook (Qwen3-Omni's code
predictor: 15 calls per step). With the whole row in one program's registers,
both filters are thresholds found by bisection and the draw is an inverse-CDF
lookup, so one launch does it.

Same distribution as the FlashInfer path (top-k first, top-p on the
renormalized survivors, temperature 0 as argmax), different random stream.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_VOCAB = 4096
# Each step halves the threshold's interval: after 20 it is max_prob / 2**20.
# Distinct probabilities from bf16 logits differ by at least ~0.4% relative, so
# only probabilities below ~1e-4 * max_prob can merge with a neighbour, and a
# merge keeps one extra near-tied token. Steps are the kernel's cost (~0.35 us).
_BISECT_STEPS = 20


@triton.jit
def _small_vocab_sample_kernel(
    logits, temperature, top_k, top_p, seed, offset, out, stride_b, V,
    BLOCK: tl.constexpr, STEPS: tl.constexpr,
):
    b = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    valid = idx < V
    x = tl.load(logits + b * stride_b + idx, mask=valid, other=float("-inf")).to(tl.float32)
    t = tl.load(temperature + b).to(tl.float32)
    k = tl.load(top_k + b).to(tl.int32)
    p_keep = tl.load(top_p + b).to(tl.float32)

    xmax = tl.max(x, 0)
    if t == 0.0:
        # greedy: the first maximal logit
        tok = tl.min(tl.where(x == xmax, idx, BLOCK), 0)
    else:
        e = tl.where(valid, tl.exp((x - xmax) / t), 0.0)
        prob = e / tl.sum(e, 0)
        # top-k: the largest threshold keeping at least k tokens (k <= 0: all)
        if (k > 0) & (k < V):
            lo = 0.0
            hi = tl.max(prob, 0)
            for _ in tl.static_range(STEPS):
                mid = (lo + hi) * 0.5
                enough = tl.sum(tl.where(prob >= mid, 1, 0), 0) >= k
                lo = tl.where(enough, mid, lo)
                hi = tl.where(enough, hi, mid)
            prob = tl.where(prob >= lo, prob, 0.0)
            prob = prob / tl.sum(prob, 0)
        # top-p: the largest threshold whose survivors still carry top_p
        if p_keep < 1.0:
            lo = 0.0
            hi = tl.max(prob, 0)
            for _ in tl.static_range(STEPS):
                mid = (lo + hi) * 0.5
                enough = tl.sum(tl.where(prob >= mid, prob, 0.0), 0) >= p_keep
                lo = tl.where(enough, mid, lo)
                hi = tl.where(enough, hi, mid)
            prob = tl.where(prob >= lo, prob, 0.0)
        cdf = tl.cumsum(prob, 0)
        u = tl.rand(tl.load(seed + b).to(tl.int32), tl.load(offset + b).to(tl.int32)) * tl.max(cdf, 0)
        # the first token whose CDF passes u, never one filtered out
        tok = tl.min(tl.where((cdf > u) & (prob > 0.0), idx, BLOCK), 0)
        tok = tl.where(tok < BLOCK, tok, tl.argmax(prob, 0))
    tl.store(out + b, tok.to(tl.int64))


def small_vocab_sample(
    logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor,
    top_p: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor,
) -> torch.Tensor:
    """One int64 token per row of ``logits (B, V <= MAX_VOCAB)``; the per-row
    parameters are ``(B,)`` device tensors, each row drawing from its own
    ``(seed, offset)`` Philox stream."""
    B, V = logits.shape
    assert V <= MAX_VOCAB and logits.stride(1) == 1
    out = torch.empty(B, dtype=torch.int64, device=logits.device)
    _small_vocab_sample_kernel[(B,)](
        logits, temperature, top_k, top_p, seed, offset, out, logits.stride(0), V,
        BLOCK=triton.next_power_of_2(V), STEPS=_BISECT_STEPS, num_warps=4,
    )
    return out
