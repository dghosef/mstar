"""`FlashInferCascadeDecodeWrapper` attends as the plain decode wrapper does.

Rows that start with the same prefix-cache pages read them once per group and merge with
their own pages' attention; the output is the plain decode kernel's at every row, whatever
the grouping, including after a replay planned for another grouping.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mstar.engine.resources.attn.workspace import Workspace
from mstar.engine.resources.attn.wrappers import (
    FlashInferCascadeDecodeWrapper,
    FlashInferDecodeWrapper,
    shared_prefix_groups,
)
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer needs a GPU")

QO_HEADS, KV_HEADS, DIM, PAGE = 32, 8, 128, 16
BS = 8
WS = 128 << 20

# per row: (prefix id or None, prefix pages, own tokens); padding rows fill the bucket
LAYOUTS = {
    # two groups; a row that starts with group a's first page only, left out of it; a row of
    # its own; a padding row
    "two_groups": [("a", 6, 37), ("b", 4, 5), ("a", 6, 16), ("a", 6, 1), ("b", 4, 70), (None, 0, 50),
                   ("a", 1, 40)],
    "none_shared": [(None, 0, 40), (None, 0, 3), (None, 0, 90)],
    "all_one_group": [("a", 5, n) for n in (1, 2, 17, 33, 16, 8, 64, 5)],
}


def _layout(rows, page_base=SINK_PAGE + 1):
    """Host index arrays as the KV plan builds them: rows with the same prefix id name the
    same first pages, then pages of their own; padding rows on the sink page."""
    prefix_pages: dict[str, list[int]] = {}
    nxt = page_base

    def fresh(n):
        nonlocal nxt
        pages = list(range(nxt, nxt + n))
        nxt += n
        return pages

    kvp, idx, last, lens = [0], [], [], []
    for pid, shared, own in rows:
        pages = []
        if pid is not None:
            if pid not in prefix_pages:
                prefix_pages[pid] = fresh(16)
            pages = prefix_pages[pid][:shared]
        n_own = -(-own // PAGE)
        pages = pages + fresh(n_own)
        idx.extend(pages)
        kvp.append(kvp[-1] + len(pages))
        last.append((own - 1) % PAGE + 1)
        lens.append((len(pages) - 1) * PAGE + last[-1])
    for _ in range(BS - len(rows)):
        idx.append(SINK_PAGE)
        kvp.append(kvp[-1] + 1)
        last.append(1)
        lens.append(1)
    def i32(v):
        return torch.tensor(v, dtype=torch.int32)

    return dict(
        paged_kv_indptr=i32(kvp), paged_kv_indices=i32(idx), paged_kv_last_page_len=i32(last),
        kv_lens=i32(lens).as_subclass(HostLens),
    ), nxt


def test_shared_prefix_groups():
    arrays, _ = _layout(LAYOUTS["two_groups"])
    kvp, idx = arrays["paged_kv_indptr"].numpy(), arrays["paged_kv_indices"].numpy()
    groups = {tuple(int(r) for r in rows): run for rows, run in shared_prefix_groups(kvp, idx, min_pages=2)}
    # row 6 shares one page with group a: it stays out rather than cut a's run to one page
    assert groups == {(0, 2, 3): 6, (1, 4): 4}
    # a row's last page is its own: identical page lists still leave each row one page
    same = np.array([0, 3, 6]), np.array([5, 6, 7, 5, 6, 7])
    assert [(list(r), n) for r, n in shared_prefix_groups(*same, min_pages=1)] == [([0, 1], 2)]
    # padding rows on the sink page share it, and are in no group
    sink = np.array([0, 2, 4]), np.array([SINK_PAGE, SINK_PAGE, SINK_PAGE, SINK_PAGE])
    assert shared_prefix_groups(*sink, min_pages=1) == []
    # a run shorter than the threshold is no group
    assert shared_prefix_groups(*same, min_pages=3) == []


@requires_gpu
def test_cascade_decode_attends_as_the_plain_decode_does():
    device = torch.device("cuda")
    torch.manual_seed(0)
    kw = dict(num_qo_heads=QO_HEADS, num_kv_heads=KV_HEADS, head_dim=DIM, page_size=PAGE, device=device)
    max_pages = 512

    def ws():
        return Workspace(torch.empty(WS, dtype=torch.uint8, device=device), "test")

    cascade = FlashInferCascadeDecodeWrapper(
        workspace_buffer=ws(), batch_size=BS, max_num_pages=max_pages, **kw,
    )
    plain = FlashInferDecodeWrapper(
        workspace_buffer=ws(), batch_size=BS, max_num_pages=max_pages,
        accelerator_graph=True, **kw,
    )
    kv = torch.randn(max_pages, 2, PAGE, KV_HEADS, DIM, dtype=torch.bfloat16, device=device)
    q = torch.randn(BS, QO_HEADS, DIM, dtype=torch.bfloat16, device=device)

    def both(name):
        arrays, _ = _layout(LAYOUTS[name])
        cascade.plan(**arrays)
        plain.plan(**arrays)

    # captured on one grouping, replayed on every other
    both("two_groups")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        cascade.run(q, kv), plain.run(q, kv)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got, want = cascade.run(q, kv), plain.run(q, kv)
    for name in ("two_groups", "none_shared", "all_one_group", "two_groups"):
        both(name)
        graph.replay()
        torch.cuda.synchronize()
        real = len(LAYOUTS[name])
        torch.testing.assert_close(got[:real].float(), want[:real].float(), atol=2e-2, rtol=2e-2)
