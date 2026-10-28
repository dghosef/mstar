"""Pre-norm transformer decoder layers.

``DecoderLayer``: the standard pre-norm block (norm → attn → residual,
norm → mlp → residual). Composes any ``nn.Module`` for the attention,
MLP, and norms — so models pick the variants they want and pass them in.

``GatedDecoderLayer``: variant for adaRMS conditioning (pi05 action
expert). The norms are ``AdaRMSNorm`` and return ``(normed, gate)``; the
residual is ``x + gate * y`` instead of ``x + y``.

``run_layers``: a stack of ``DecoderLayer``s with each residual add deferred
into the next norm (``RMSNorm.forward_residual``).
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import nn


def _gated_residual(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None,
) -> torch.Tensor:
    """``x + gate * y`` with a None-gate fallback to plain addition."""
    if gate is None:
        return x + y
    return x + y * gate


class DecoderLayer(nn.Module):
    """Standard pre-norm transformer decoder layer.

    Computes:
        residual = x
        x = input_layernorm(x); x = self_attn(x); x = residual + x
        residual = x
        x = post_attention_layernorm(x); x = mlp(x); x = residual + x

    Args:
        self_attn: attention module taking ``(hidden_states, label=,
            layer_idx=)`` and returning a tensor.
        mlp: feedforward module taking and returning a tensor.
        input_layernorm: pre-attention norm.
        post_attention_layernorm: pre-FFN norm.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        mlp: nn.Module,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_residual(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None,
        attend: Callable[[torch.Tensor], torch.Tensor] | None = None,
        comm_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer with its residual adds left to the norms: returns the MLP
        output and the stream it has not been added to yet. ``attend`` stands in
        for ``self_attn`` on the normed input (a code predictor's own kernels).
        Under TP the norms reduce the row-parallel partial sums over
        ``comm_group``."""
        h, residual = self.input_layernorm.forward_residual(hidden_states, residual, comm_group)
        h = self.self_attn(h) if attend is None else attend(h)
        h, residual = self.post_attention_layernorm.forward_residual(h, residual, comm_group)
        return self.mlp(h), residual


def run_layers(
    layers: nn.ModuleList, hidden_states: torch.Tensor, norm: nn.Module,
    attend: Callable[[int, nn.Module, torch.Tensor], torch.Tensor] | None = None,
    comm_group=None,
) -> torch.Tensor:
    """``layers`` then the final ``norm``, each residual add fused into the next
    norm (one add+norm kernel instead of two launches; under TP, with
    ``comm_group``, all-reduce+add+norm). The stream starts at ``hidden_states``,
    a full tensor, and later norms update it in place, so pass a tensor the
    caller may overwrite.

    By default layer ``i`` attends through its engine attention, its layer
    cursor moved to ``i`` first; ``attend(i, self_attn, h)`` replaces that.
    """
    residual = None
    for i, layer in enumerate(layers):
        if attend is None:
            layer.self_attn.attend.set_layer_idx(i)
            hidden_states, residual = layer.forward_residual(hidden_states, residual, comm_group=comm_group)
        else:
            hidden_states, residual = layer.forward_residual(
                hidden_states, residual, partial(attend, i, layer.self_attn), comm_group,
            )
    normed, _ = norm.forward_residual(hidden_states, residual, comm_group)
    return normed


class GatedDecoderLayer(nn.Module):
    """Pre-norm decoder layer with adaRMS gated residuals.

    The norms must be ``AdaRMSNorm``-shaped: ``forward(x, cond)`` returns
    ``(normed, gate)``. The residual becomes ``x + gate * y``.

    Used by pi05's action expert; ``adarms_cond`` is the shared condition
    vector consumed by both norms.
    """

    def __init__(
        self,
        self_attn: nn.Module,
        mlp: nn.Module,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self, hidden_states: torch.Tensor, adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        normed, gate = self.input_layernorm(hidden_states, adarms_cond)
        attn_out = self.self_attn(normed)
        hidden_states = _gated_residual(residual, attn_out, gate)

        residual = hidden_states
        normed, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        mlp_out = self.mlp(normed)
        return _gated_residual(residual, mlp_out, gate)
