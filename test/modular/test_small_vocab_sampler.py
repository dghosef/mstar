"""The one-kernel small-vocabulary sampler against the distribution it targets (GPU only)."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _target(row: torch.Tensor, temp: float, top_k: int, top_p: float) -> torch.Tensor:
    """softmax / temperature, top-k, renormalize, smallest top-p set, renormalize."""
    pr = torch.softmax(row.float() / temp, -1)
    if 0 < top_k < row.numel():
        pr = torch.where(pr >= pr.topk(top_k).values[-1], pr, 0)
        pr = pr / pr.sum()
    if top_p < 1:
        sv, si = pr.sort(descending=True)
        keep = torch.zeros_like(pr)
        keep[si[(sv.cumsum(0) - sv) < top_p]] = 1
        pr = pr * keep
    return pr / pr.sum()


@pytest.mark.parametrize("temp,top_k,top_p", [(1.0, 50, 0.8), (0.9, 50, 1.0), (1.0, 0, 0.9), (0.7, 20, 0.95)])
def test_draws_follow_the_filtered_distribution(temp, top_k, top_p):
    from mstar.engine.resources.sampler.small_vocab import small_vocab_sample

    torch.manual_seed(0)
    n, v = 100_000, 2048
    row = (torch.randn(v, device="cuda") * 2.5).to(torch.bfloat16)
    tokens = small_vocab_sample(
        row.expand(n, v).contiguous(),
        torch.full((n,), temp, device="cuda"),
        torch.full((n,), top_k, device="cuda", dtype=torch.int32),
        torch.full((n,), top_p, device="cuda"),
        torch.arange(n, device="cuda") * 7 + 1,
        torch.arange(n, device="cuda"),
    )
    target = _target(row, temp, top_k, top_p)
    seen = torch.bincount(tokens, minlength=v).float() / n
    assert (seen[target == 0] == 0).all(), "drew a filtered-out token"
    # total variation at the sampling-noise level for 1e5 draws
    assert 0.5 * (seen - target).abs().sum().item() < 0.02


def test_temperature_zero_is_argmax_and_rows_are_reproducible():
    from mstar.engine.resources.sampler.small_vocab import small_vocab_sample

    logits = torch.randn(64, 3000, device="cuda").to(torch.bfloat16)
    args = (torch.full((64,), 50, device="cuda", dtype=torch.int32), torch.full((64,), 0.8, device="cuda"),
            torch.arange(64, device="cuda"), torch.zeros(64, device="cuda", dtype=torch.int64))
    greedy = small_vocab_sample(logits, torch.zeros(64, device="cuda"), *args)
    assert torch.equal(greedy, logits.float().argmax(-1))
    once = small_vocab_sample(logits, torch.ones(64, device="cuda"), *args)
    # a row's draw depends on its own (seed, offset), not its position in the batch
    flipped = small_vocab_sample(logits.flip(0), torch.ones(64, device="cuda"), *(a.flip(0) for a in args))
    assert torch.equal(once, flipped.flip(0))
