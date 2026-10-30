"""Joint attention over a DiT's packed tokens.

A diffusion transformer attends bidirectionally over all of a request's tokens
(text + image + reference-image) and keeps nothing between steps, so there is
no KV cache to page. Three backends serve the same ``[B, S, H, D]`` contract:

* ``sdpa``   ``torch.nn.functional.scaled_dot_product_attention`` — the reference
             pipelines' own kernel, hence the parity baseline; used eagerly and on CPU.
* ``fa3``    FlashAttention-3's dense kernel (Hopper) on the same unpadded batch — no
             planning, and a traced op inside the compiled transformer. On H100 it runs
             0.78-0.93x the time of cuDNN's SDPA at DiT shapes (4k-5k tokens, 128-dim heads).
* the engine's **ragged attention resource** (``RaggedAttentionSpec``, FlashInfer's
  ragged prefill on FA2/FA3) — one segment per request, planned by the runner
  outside the captured graph, so a per-step CUDA graph can replay it. The
  resource is bound by the submodule and threaded to the layers as a callable.

Layers call :func:`joint_attention`; the choice is made once per forward by
whoever owns the resource, not per layer.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F

try:  # the wheel the engine's dense attention uses too; Hopper only
    from fa3_fwd_interface import flash_attn_func as _fa3_flash_attn_func
except Exception:  # noqa: BLE001
    _fa3_flash_attn_func = None

# ``(q, k, v) -> out`` over packed ``[total_tokens, H, D]`` tensors, one segment per request.
RaggedAttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class DenseAttention(NamedTuple):
    """A dense kernel over ``[B, S, H, D]`` q / k / v that a submodule threads to its layers
    where a ragged resource's callable would go, in place of SDPA."""

    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def fa3_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bidirectional FlashAttention-3 over ``[B, S, H, D]``, its native layout (no transposes)."""
    out = _fa3_flash_attn_func(q, k, v, causal=False)
    return out[0] if isinstance(out, tuple) else out


FA3_DENSE = DenseAttention(fa3_attention)


def dense_attention_for(backend: str, device: torch.device | str | None = None) -> DenseAttention | None:
    """The dense kernel an ``attention_backend`` names (None for ``sdpa`` and ``flashinfer``),
    refusing ``fa3`` where it cannot run rather than falling back silently."""
    if backend != "fa3":
        return None
    if _fa3_flash_attn_func is None:
        raise ValueError("attention_backend 'fa3' needs the fa3-fwd wheel (fa3_fwd_interface), which is not installed")
    if torch.cuda.is_available() and torch.device(device or "cuda").type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) != (9, 0):
            raise ValueError(f"attention_backend 'fa3' runs on Hopper (sm90) only; this GPU is sm{major}{minor}")
    return FA3_DENSE


class RaggedAttentionMixin:
    """:meth:`ragged_for` for a ``NodeSubmodule`` whose layers attend through the ragged
    attention resource named ``attn_resource_key``, or through ``dense_attention``
    when it has none (None for both: SDPA everywhere)."""

    attn_resource_key: str | None = None
    dense_attention: DenseAttention | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One callable per attention label for the life of a binding.
        self._ragged_fns: dict[str, RaggedAttentionFn] = {}

    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._ragged_fns.clear()  # the memoized callables hold the old resource

    def ragged_for(self, label: str) -> RaggedAttentionFn | DenseAttention | None:
        """``(q, k, v) -> out`` over the segments declared under ``label``; without a
        ragged resource, the submodule's ``dense_attention`` (any label: a dense kernel
        needs no declared segments) or None (layers then fall back to SDPA).

        The resource's ``run`` with the label bound, built once per label for the life
        of the binding rather than per step."""
        if self.attn_resource_key is None:
            return self.dense_attention
        if label not in self._ragged_fns:
            resource = self.node_resources.get(self.attn_resource_key)
            if resource is None:
                return None
            self._ragged_fns[label] = functools.partial(resource.run, label=label)
        return self._ragged_fns[label]


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bidirectional attention over ``[B, S, H, D]`` tensors via SDPA (``k`` / ``v`` may
    have their own length); returns ``[B, S, H, D]``."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False,
    )
    return out.transpose(1, 2)


def joint_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ragged: RaggedAttentionFn | DenseAttention | None = None,
) -> torch.Tensor:
    """Attention of ``[B, S, H, D]`` queries over ``[B, S_kv, H_kv, D]`` keys / values
    (all requests in the batch share ``S`` and ``S_kv``; ``S_kv == S`` for
    self-attention) through the ragged resource when one is bound, a dense kernel
    the submodule chose, else SDPA.

    The ragged kernel wants the batch packed to ``[B * S, H, D]`` / ``[B * S_kv, ...]``
    with one segment per request — exactly what the owning submodule declared in its
    step (a cross-attention pair for ``S_kv != S``) — and hands back the query packing.
    """
    if ragged is None:
        return sdpa_attention(q, k, v)
    if isinstance(ragged, DenseAttention):
        return ragged.fn(q, k, v)
    bsz, seq, heads, dim = q.shape
    kv_seq, kv_heads = k.shape[1], k.shape[2]
    out = ragged(
        q.reshape(bsz * seq, heads, dim),
        k.reshape(bsz * kv_seq, kv_heads, dim),
        v.reshape(bsz * kv_seq, kv_heads, dim),
    )
    return out.view(bsz, seq, heads, dim).to(q.dtype)
