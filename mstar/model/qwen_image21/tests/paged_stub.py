"""A CPU stand-in for the paged KV cache and its attention, for tests.

It keeps each ``(request, label)`` stream's committed K/V per layer and attends a step's rows over
them with SDPA: a causal step's queries see everything before them plus a bottom-right triangle over
their own keys (FlashInfer's causal mask over a paged prefix), a non-causal step's see all of it.
Driven either row by row (``step``) or from a declared ``SubmoduleStep`` (``begin`` / ``bind_step``
/ ``end``), whose segments, causality, commit, combined labels and post-forks it follows.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class Row(NamedTuple):
    stream: tuple  # (request, label)
    length: int


class PagedAttendStub:
    def __init__(self, kv_key: str = "kv", attn_key: str = "attn"):
        self.kv_key, self.attn_key = kv_key, attn_key
        self.cache: dict[tuple, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.rows: list[Row] = []
        self.causal = False
        self.commit = False
        self.layer = 0
        self._declared = None

    def step(self, rows: list[Row], causal: bool, commit: bool) -> "PagedAttendStub":
        self.rows, self.causal, self.commit = list(rows), causal, commit
        return self

    # ------------------------------------------------------------ declared steps
    def begin(self, declared) -> None:
        self._declared = declared
        kv, attn = declared.steps[self.kv_key], declared.steps[self.attn_key]
        assert [tuple(s) for s in kv.segments] == [tuple(s) for s in attn.segments]
        self.causal, self.commit = attn.causal, kv.commit

    def bind_step(self, label: str) -> None:
        kv = self._declared.steps[self.kv_key]
        members = next((labels for labels, name in kv.combined_labels.items() if name == label), (label,))
        self.rows = [Row((s.request_id, s.label), s.span) for s in kv.segments if s.label in members]

    def end(self, declared) -> None:
        kv = declared.steps[self.kv_key]
        for src, dst in kv.post_forks:
            for rid in kv.fork_rids if kv.fork_rids is not None else {s.request_id for s in kv.segments}:
                self.fork((rid, src), (rid, dst))
        self._declared = None

    def fork(self, src: tuple, dst: tuple) -> None:
        self.cache[dst] = {layer: (k.clone(), v.clone()) for layer, (k, v) in self.cache.get(src, {}).items()}

    def set_layer_idx(self, layer_idx: int) -> None:
        self.layer = layer_idx

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out, offset = [], 0
        for row in self.rows:
            qs, ks, vs = (t[offset:offset + row.length] for t in (q, k, v))
            offset += row.length
            stored = self.cache.setdefault(row.stream, {}).get(self.layer)
            if stored is not None:
                ks, vs = torch.cat([stored[0], ks]), torch.cat([stored[1], vs])
            if self.commit:
                self.cache[row.stream][self.layer] = (ks, vs)
            mask = None
            if self.causal:
                n, total = qs.shape[0], ks.shape[0]
                mask = torch.ones(n, total, dtype=torch.bool, device=qs.device).tril(total - n)
            attn = F.scaled_dot_product_attention(
                qs.transpose(0, 1)[None], ks.transpose(0, 1)[None], vs.transpose(0, 1)[None], attn_mask=mask,
            )
            out.append(attn[0].transpose(0, 1))
        assert offset == q.shape[0], (offset, q.shape)
        return torch.cat(out)
