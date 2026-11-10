"""A padded prefill must not hand the chunked kernel a zero-length segment.

The GDN prefill kernel is warp-specialised: a row covering no tokens leaves its
consumers waiting on a barrier its producer never reaches, so the launch never
retires and every rank of a TP instance wedges behind it — no error, no log.
A replay's padding rows arrive with span 0 (`PackedCudaGraphConfig` pads with
`distribute_tokens(0, n)`), which is exactly that shape.

CPU-only: nothing here launches a kernel.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.linear_attn.wrappers import GDNPrefillWrapper


def build_wrapper(bs: int, num_tokens: int, **kwargs) -> GDNPrefillWrapper:
    return GDNPrefillWrapper(
        device=torch.device("cpu"),
        pad_slot_id=0,
        num_tokens=num_tokens,
        bs=bs,
        accelerator_graph=True,
        **kwargs,
    )


def plan_spans(wrapper: GDNPrefillWrapper, spans: list[int]):
    n = len(spans)
    wrapper.plan(
        spans,
        slots=torch.zeros(n, dtype=torch.int32),
        has_state=torch.zeros(n, dtype=torch.bool),
    )
    return wrapper._plan_state


def test_padding_rows_get_a_token_of_their_own():
    """Three real rows in a bucket of four: every segment is non-empty."""
    state = plan_spans(build_wrapper(bs=4, num_tokens=1024), [300, 300, 232, 0])
    cu = state.cu_seqlens.tolist()
    assert cu == [0, 300, 600, 832, 833]
    assert all(b > a for a, b in zip(cu, cu[1:], strict=False)), cu


def test_no_empty_segment_even_when_the_batch_fills_the_bucket():
    """The borrowed tokens come from room `run` adds, not from the bucket, so
    a batch that fills its token budget exactly is still safe."""
    spans = [256, 256, 256, 256] + [0] * 12
    state = plan_spans(build_wrapper(bs=16, num_tokens=1024), spans)
    cu = state.cu_seqlens.tolist()
    assert cu[-1] == 1024 + 12
    assert all(b > a for a, b in zip(cu, cu[1:], strict=False)), cu


def test_borrowed_tokens_stay_masked_off():
    """The mask still marks only the real tokens, so `run` zeroes what the
    padding rows read."""
    state = plan_spans(build_wrapper(bs=4, num_tokens=1024), [300, 300, 232, 0])
    assert state.token_mask[:832].all()
    assert not state.token_mask[832:].any()


def test_a_batch_that_needs_no_padding_is_untouched():
    state = plan_spans(build_wrapper(bs=4, num_tokens=1024), [256, 256, 256, 256])
    assert state.cu_seqlens.tolist() == [0, 256, 512, 768, 1024]


def test_eager_step_borrows_only_what_it_needs():
    """An eager step has no bucket tail to spend, so `run` pads by exactly the
    number of rows the fixup lengthened — never past the tokens it was given."""
    w = GDNPrefillWrapper(device=torch.device("cpu"), pad_slot_id=0, accelerator_graph=False)
    state = plan_spans(w, [128, 0, 64])
    assert w._plan_borrowed == 1
    cu = state.cu_seqlens.tolist()
    assert cu == [0, 128, 129, 193]
    assert all(b > a for a, b in zip(cu, cu[1:], strict=False)), cu


@pytest.mark.parametrize("per_row, fits", [(2, False), (3, True)])
def test_a_captured_step_has_room_for_the_snapshots_per_row_its_policy_keeps(per_row, fits):
    """A graph is captured for a number of snapshots, a bound per row of its bucket. A row whose policy keeps a third
    (a message boundary beside the row's end and the KV hit's) needs the capture to have counted it."""
    every = 128
    w = build_wrapper(bs=1, num_tokens=1024, **({} if per_row == 2 else {"snapshots_per_row": per_row}))
    snapshots = [(0, offset, slot) for slot, offset in enumerate((128, 256, 384), 1)]
    plan = lambda: w.plan(  # noqa: E731
        [512], slots=torch.zeros(1, dtype=torch.int32), has_state=torch.zeros(1, dtype=torch.bool),
        snapshots=snapshots, every=every, checkpoints=torch.zeros(1024 // every + 2, 4, 4))
    if fits:
        plan()
        assert w._num_snapshots == per_row
    else:
        with pytest.raises(AssertionError, match="3 snapshots in a step captured for 2"):
            plan()
