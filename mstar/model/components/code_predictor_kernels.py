"""Kernels of a residual code predictor's depth loop (Qwen3-Omni's and
Qwen3-TTS's): a few layers walked once per codec group, 16 x 5 times per Talker
step, over a dense frame-local KV cache.

``cp_attend`` is a layer's attention between its projections in one launch:
q/k norm, RoPE, the cache write and attention over the depth positions so far,
each a few microseconds around a ~1 us job when launched apart.
``cp_embed_accumulate`` records a depth step's codes and sums their embeddings.
``DepthAttention`` is a code predictor layer's attention built on ``cp_attend``,
over the frame-local KV cache its ``DepthKV`` holds.

A captured forward compiles with static ints, so any compiled frame that holds a
depth step's ``cache_pos`` or a layer index recompiles at every depth position
or layer. Both live on the ``DepthKV``, which the depth loop binds per step and
``run_layers`` moves per layer, and are read only in its eager methods: the
layers' frames, the projections included, compile once per shape.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.model.components.distributed.attention import ParallelAttention
from mstar.utils.attention import rms_norm_rope


@triton.jit(do_not_specialize=["cache_pos"])
def _cp_attend_kernel(
    QKV, Wq, Wk, Pos, KV, O,
    eps_q, eps_k, rope_theta, sm_scale, cache_pos,
    s_qkv, s_o,
    s_kv_b, s_kv_c, s_kv_n, s_kv_h,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
    SEQ: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One (token, q head) of a depth step: its q normed and roped, the step's
    keys up to it normed and roped from ``QKV`` (the step's own k/v are used
    from registers, so no program waits on another's cache write), the
    earlier positions from the cache, and attention over all of them. The
    first q head of each kv head also writes its token's k and v into the
    cache for later steps."""
    m = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // (H_Q // H_KV)
    b = m // SEQ
    s = m % SEQ
    elem = KV.dtype.element_ty
    offs = tl.arange(0, HALF_D)
    d = tl.arange(0, D)
    row = QKV + m * s_qkv
    pos = tl.load(Pos + m).to(tl.float32)
    q1, q2 = rms_norm_rope(
        tl.load(row + h * D + offs).to(tl.float32), tl.load(row + h * D + HALF_D + offs).to(tl.float32),
        Wq, pos, eps_q, rope_theta, D, HALF_D,
    )
    # rounded where the cache and the unfused q round
    q1 = q1.to(elem).to(tl.float32) * sm_scale
    q2 = q2.to(elem).to(tl.float32) * sm_scale

    n = tl.arange(0, BLOCK_N)
    kv_base = KV + b * s_kv_b + hk * s_kv_h
    cached = n < cache_pos
    k_off = n[:, None] * s_kv_n + offs[None, :]
    k1 = tl.load(kv_base + k_off, mask=cached[:, None], other=0.0).to(tl.float32)
    k2 = tl.load(kv_base + k_off + HALF_D, mask=cached[:, None], other=0.0).to(tl.float32)
    v = tl.load(kv_base + s_kv_c + n[:, None] * s_kv_n + d[None, :], mask=cached[:, None], other=0.0).to(tl.float32)
    for j in tl.static_range(SEQ):
        if j <= s:
            rj = QKV + (b * SEQ + j) * s_qkv
            pj = tl.load(Pos + b * SEQ + j).to(tl.float32)
            kj1, kj2 = rms_norm_rope(
                tl.load(rj + (H_Q + hk) * D + offs).to(tl.float32),
                tl.load(rj + (H_Q + hk) * D + HALF_D + offs).to(tl.float32),
                Wk, pj, eps_k, rope_theta, D, HALF_D,
            )
            kj1 = kj1.to(elem)
            kj2 = kj2.to(elem)
            vj = tl.load(rj + (H_Q + H_KV + hk) * D + d).to(elem)
            if j == s:
                if h % (H_Q // H_KV) == 0:
                    dst = kv_base + (cache_pos + s) * s_kv_n
                    tl.store(dst + offs, kj1)
                    tl.store(dst + HALF_D + offs, kj2)
                    tl.store(dst + s_kv_c + d, vj)
            here = (n == cache_pos + j)[:, None]
            k1 = tl.where(here, kj1.to(tl.float32)[None, :], k1)
            k2 = tl.where(here, kj2.to(tl.float32)[None, :], k2)
            v = tl.where(here, vj.to(tl.float32)[None, :], v)
    valid = n <= cache_pos + s  # causal within the depth sequence
    qk = tl.where(valid, tl.sum(k1 * q1[None, :], 1) + tl.sum(k2 * q2[None, :], 1), float("-inf"))
    p = tl.exp(qk - tl.max(qk, 0))
    out = tl.sum(p[:, None] * v, 0) / tl.sum(p, 0)
    tl.store(O + m * s_o + h * D + d, out.to(O.dtype.element_ty))


@torch.compiler.disable
def cp_attend(
    qkv: torch.Tensor, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, pos: torch.Tensor,
    kv_layer: torch.Tensor, cache_pos: int, seq_len: int, eps_q: float, eps_k: float,
    rope_theta: float, n_q_heads: int, n_kv_heads: int, head_dim: int,
) -> torch.Tensor:
    """A code predictor layer's attention between its projections, for
    ``qkv (tokens, (Hq + 2 Hkv) * D)``, token ``b * seq_len + s`` at depth
    position ``cache_pos + s`` (``pos`` holds its RoPE position): q/k RMSNorm,
    RoPE, the new k and v written into ``kv_layer (bs, 2, S, Hkv, D)``, and
    causal attention over the depth positions so far. Returns
    ``(tokens, Hq * D)``. One launch, one warp per (token, head): the depth
    sequence is at most 16 long, so a warp holds its whole key range."""
    tokens = qkv.shape[0]
    assert qkv.stride(1) == 1 and kv_layer.stride(-1) == 1 and tokens % seq_len == 0
    out = torch.empty(tokens, n_q_heads * head_dim, device=qkv.device, dtype=qkv.dtype)
    _cp_attend_kernel[(tokens, n_q_heads)](
        qkv, q_norm_w, k_norm_w, pos, kv_layer, out,
        eps_q, eps_k, rope_theta, head_dim ** -0.5, cache_pos,
        qkv.stride(0), out.stride(0),
        kv_layer.stride(0), kv_layer.stride(1), kv_layer.stride(2), kv_layer.stride(3),
        H_Q=n_q_heads, H_KV=n_kv_heads, D=head_dim, HALF_D=head_dim // 2,
        SEQ=seq_len, BLOCK_N=triton.next_power_of_2(kv_layer.shape[2]),
        num_warps=1, num_stages=1,
    )
    return out


@triton.jit
def _cp_embed_accumulate_kernel(
    Tok, W, P, Emb, Sum, Codes, group, s_w, s_p, s_emb, s_sum, s_codes,
    H: tl.constexpr, HP: tl.constexpr, BLOCK: tl.constexpr, PROJ: tl.constexpr, INIT: tl.constexpr,
):
    b = tl.program_id(0)
    tok = tl.load(Tok + b)
    offs = tl.arange(0, BLOCK)
    for h0 in tl.static_range(0, H, BLOCK):
        e = tl.load(W + tok * s_w + h0 + offs)
        if not PROJ:
            tl.store(Emb + b * s_emb + h0 + offs, e)
        if INIT:
            tl.store(Sum + b * s_sum + h0 + offs, e.to(Sum.dtype.element_ty))
        else:
            acc = tl.load(Sum + b * s_sum + h0 + offs)
            tl.store(Sum + b * s_sum + h0 + offs, acc + e.to(acc.dtype))
    if PROJ:
        for h0 in tl.static_range(0, HP, BLOCK):
            tl.store(Emb + b * s_emb + h0 + offs, tl.load(P + tok * s_p + h0 + offs))
    tl.store(Codes + b * s_codes + group, tok.to(Codes.dtype.element_ty))


@torch.compiler.disable
def cp_embed_accumulate(
    tokens: torch.Tensor, weight: torch.Tensor, emb_sum: torch.Tensor,
    all_codes: torch.Tensor, group: int, inputs: torch.Tensor | None = None,
    init: bool = False,
) -> torch.Tensor:
    """For a depth step's sampled ``tokens (bs,)``: record them as ``all_codes[:, group]``,
    look up their embeddings in ``weight (V, H)`` and add those to ``emb_sum (bs, H)`` in
    its dtype (or, with ``init``, start the sum from them). Returns the next depth step's
    input: the embeddings, or their rows of ``inputs (V, H')`` when given (the embedding
    table already projected to the predictor's width). One launch for what was an
    index_select, a copy, a promoting add and the input projection."""
    bs, hidden = tokens.shape[0], weight.shape[1]
    table = weight if inputs is None else inputs
    assert weight.stride(1) == 1 and table.stride(1) == 1 and emb_sum.stride(1) == 1
    assert hidden % 256 == 0 and table.shape[1] % 256 == 0
    emb = torch.empty(bs, table.shape[1], dtype=table.dtype, device=weight.device)
    _cp_embed_accumulate_kernel[(bs,)](
        tokens, weight, table, emb, emb_sum, all_codes, group,
        weight.stride(0), table.stride(0), emb.stride(0), emb_sum.stride(0), all_codes.stride(0),
        H=hidden, HP=table.shape[1], BLOCK=256, PROJ=inputs is not None, INIT=init, num_warps=4,
    )
    return emb


class DepthKV:
    """A code predictor's frame-local KV cache and the depth step attending
    over it, as its layers' attention cursor: the depth loop calls
    ``bind_step`` per step, and ``run_layers`` calls ``set_layer_idx`` per
    layer. Its methods are all eager, so the ints stay out of compiled frames
    (and none is ``recursive=False``; see ``run_layers``). A binding serves
    one pass over the layers: the pass's first layer consumes it, so a step
    that was not bound fails instead of reusing the previous step's slots."""

    def __init__(self) -> None:
        self._fresh = False

    @torch.compiler.disable
    def bind_step(self, kv_cache: torch.Tensor, pos: torch.Tensor, cache_pos: int) -> None:
        """Bind the next pass over the layers.

        Args:
            kv_cache: ``[n_layers, bs, 2, max_seq_len, n_kv_heads, head_dim]``
                dense tensor preallocated by the submodule. It is not cleared between
                frames: every position a depth step reads was written earlier in the
                same frame.
            pos: ``[bs, seq_len]`` int tensor with the absolute position of each
                input token within the depth sequence.
            cache_pos: write offset into ``kv_cache``'s seq-len axis. The new
                K/V tokens land at ``[cache_pos : cache_pos + seq_len]``.
        """
        self.kv_cache, self.pos, self.cache_pos = kv_cache, pos.reshape(-1), cache_pos
        self._fresh = True

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        if layer_idx == 0:
            if not self._fresh:
                raise ValueError("DepthKV: bind_step before each pass over the code predictor's layers")
            self._fresh = False
        self.layer_idx = layer_idx

    @torch.compiler.disable
    def __call__(self, attn: "DepthAttention", qkv: torch.Tensor) -> torch.Tensor:
        """The current layer's attention for ``qkv (bs, S, (Hq + 2 Hkv) * D)``:
        k and v written at the step's depth positions, each token attending
        over the positions so far. Returns ``(bs, S, Hq * D)``."""
        bs, seq_len, _ = qkv.shape
        return cp_attend(
            qkv.view(bs * seq_len, -1), attn.q_norm.weight, attn.k_norm.weight,
            self.pos, self.kv_cache[self.layer_idx], self.cache_pos, seq_len,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            float(attn.rope_theta), attn.num_heads, attn.num_kv_heads, attn.head_dim,
        ).view(bs, seq_len, -1)


class DepthAttention(ParallelAttention):
    """A code predictor layer's attention, ``(bs, S, hidden)`` in and out:
    the qkv and o projections around its ``DepthKV``. The projections compile
    (at -O1 a one-row GEMM becomes a fused Triton reduction); the kernels
    between them run eager."""

    def __init__(self, depth: DepthKV, **kwargs):
        super().__init__(qk_norm=True, **kwargs)
        self.attend = depth

    def bind_resources(self, resources: dict) -> None:
        """It attends through its ``DepthKV``, not the node's resources."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.attend(self, self.qkv_proj(hidden_states)))
