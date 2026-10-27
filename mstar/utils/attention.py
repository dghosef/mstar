import torch
import triton
import triton.language as tl

# ============================================================================
# RoPE from position ids (no QK-norm, in-place).
#
# Like ``flashinfer.rope.apply_rope_pos_ids_inplace`` but works for any dtype
# (in particular fp32, which flashinfer rejects). HF/Llama "rotate_half"
# layout: head_dim is split into [first_half, second_half] and rotated as
#     out_first  = x_first  * cos - x_second * sin
#     out_second = x_second * cos + x_first  * sin
# matching ``transformers.models.qwen3_omni_moe.apply_rotary_pos_emb``.
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["D"],
    # In-place kernel: restore X_ptr between autotune trials so we benchmark
    # the same input each time and don't permanently corrupt the user's tensor.
    restore_value=["X_ptr"],
)
@triton.jit
def _rope_pos_ids_kernel(
    X_ptr, Pos_ptr,
    rope_theta,
    sx_m, sx_h, sx_d,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = X_ptr + pid_m * sx_m + pid_h * sx_h

    offs_half = tl.arange(0, HALF_D)

    x_first  = tl.load(base + offs_half * sx_d).to(tl.float32)
    x_second = tl.load(base + (offs_half + HALF_D) * sx_d).to(tl.float32)

    pos = tl.load(Pos_ptr + pid_m).to(tl.float32)
    inv_freq = tl.exp(-tl.log(rope_theta) * (offs_half.to(tl.float32) * 2.0 / D))
    angle = pos * inv_freq
    cos = tl.cos(angle)
    sin = tl.sin(angle)

    out_first  = x_first  * cos - x_second * sin
    out_second = x_second * cos + x_first  * sin

    elem_ty = X_ptr.dtype.element_ty
    tl.store(base + offs_half * sx_d,            out_first.to(elem_ty))
    tl.store(base + (offs_half + HALF_D) * sx_d, out_second.to(elem_ty))


@torch.compiler.disable
def apply_rope_pos_ids(q, k, position_ids, rope_theta=10000.0):
    """In-place RoPE on q and k driven by per-token position ids.

    q:            (M, H_q,  D)  any dtype (fp32/bf16/fp16)
    k:            (M, H_kv, D)  any dtype
    position_ids: (M,) integer

    Modifies q and k in place; returns (q, k) for chaining.
    """
    M, H_q, D = q.shape
    Mk, H_kv, Dk = k.shape
    assert M == Mk, f"q and k must have same M, got {M} vs {Mk}"
    assert D == Dk, f"q and k must have same head_dim, got {D} vs {Dk}"
    assert D % 2 == 0
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert position_ids.shape == (M,)

    pos32 = position_ids.to(torch.int32)

    _rope_pos_ids_kernel[(M, H_q)](
        q, pos32,
        float(rope_theta),
        q.stride(0), q.stride(1), q.stride(2),
        D=D, HALF_D=D // 2,
    )
    _rope_pos_ids_kernel[(M, H_kv)](
        k, pos32,
        float(rope_theta),
        k.stride(0), k.stride(1), k.stride(2),
        D=D, HALF_D=D // 2,
    )
    return q, k


# ============================================================================
# Causal sliding-window self-attention (prefill).
#
# Drop-in replacement for
#   flash_attn_func(q, k, v, causal=True, window_size=(W-1, 0), softmax_scale=s)
# with q/k/v laid out as (B, S, H, D).
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=2),
    ],
    key=["S", "D", "WINDOW"],
)
@triton.jit
def _swa_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    sq_b, sq_s, sq_h, sq_d,
    sk_b, sk_s, sk_h, sk_d,
    sv_b, sv_s, sv_h, sv_d,
    so_b, so_s, so_h, so_d,
    S,
    H_q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H_q
    pid_hq = pid_bh % H_q
    pid_hkv = pid_hq // GROUP

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = (
        Q_ptr + pid_b * sq_b + offs_m[:, None] * sq_s
        + pid_hq * sq_h + offs_d[None, :] * sq_d
    )
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Causal SWA: query at row i attends to k positions
    # [max(0, i - WINDOW + 1), i]. The block-wide bounds are derived from
    # the first/last query in this BLOCK_M.
    first_q = pid_m * BLOCK_M
    last_q = first_q + BLOCK_M - 1
    first_k = tl.maximum(0, first_q - WINDOW + 1)
    # Align down to BLOCK_N so the loop steps line up; out-of-window keys
    # in the leading partial block are masked to -inf below.
    first_k_aligned = (first_k // BLOCK_N) * BLOCK_N
    last_k = tl.minimum(last_q, S - 1)

    for start_n in range(first_k_aligned, last_k + 1, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < S

        k_ptrs = (
            K_ptr + pid_b * sk_b + offs_n[:, None] * sk_s
            + pid_hkv * sk_h + offs_d[None, :] * sk_d
        )
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)

        # tl.dot: bf16/fp16 inputs accumulate in fp32 (uses tensor cores).
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        q_pos = offs_m[:, None]
        k_pos = offs_n[None, :]
        valid = (k_pos <= q_pos) & (k_pos >= q_pos - WINDOW + 1) & (k_pos < S)
        qk = tl.where(valid, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # When the entire row is still -inf (no valid keys seen yet), m_i and
        # m_new are both -inf; (-inf) - (-inf) is NaN. Substitute 0 just for
        # the arithmetic — alpha collapses to 0 and p collapses to 0 in those
        # rows, so acc/l_i correctly stay 0 until a valid key shows up.
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])

        v_ptrs = (
            V_ptr + pid_b * sv_b + offs_n[:, None] * sv_s
            + pid_hkv * sv_h + offs_d[None, :] * sv_d
        )
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = (
        O_ptr + pid_b * so_b + offs_m[:, None] * so_s
        + pid_hq * so_h + offs_d[None, :] * so_d
    )
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=q_mask)


@torch.compiler.disable
def sliding_window_attn(q, k, v, window, scale=None):
    """Causal sliding-window multi-head self-attention.

    q: (B, S, H_q, D)
    k: (B, S, H_kv, D)
    v: (B, S, H_kv, D)

    Query at position i attends to keys/values at [max(0, i - window + 1), i].
    Equivalent to ``flash_attn_func(q, k, v, causal=True,
    window_size=(window - 1, 0), softmax_scale=sm_scale)``.
    """
    B, S, H_q, D = q.shape
    _, _, H_kv, _ = k.shape
    assert H_q % H_kv == 0, "H_q must be divisible by H_kv"
    assert v.shape == (B, S, H_kv, D)
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert D in (16, 32, 64, 128, 256), f"head_dim {D} should be a power of 2 in [16,256]"
    assert window >= 1

    if scale is None:
        scale = D ** -0.5

    out = torch.empty_like(q)
    def grid(meta):
        return (triton.cdiv(S, meta["BLOCK_M"]), B * H_q)
    _swa_attn_kernel[grid](
        q, k, v, out,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S,
        H_q=H_q, GROUP=H_q // H_kv, D=D, WINDOW=window,
    )
    return out


# ============================================================================
# RMSNorm.
#
# Drop-in replacement for ``flashinfer.norm.rmsnorm(x, weight, eps)`` on
# (M, D) input. Output dtype matches input dtype; reduction in fp32.
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["D"],
)
@triton.jit
def _rms_norm_kernel(
    X_ptr, W_ptr, O_ptr,
    eps,
    sx_m, sx_d,
    so_m, so_d,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, D)

    x = tl.load(X_ptr + pid * sx_m + offs * sx_d).to(tl.float32)
    w = tl.load(W_ptr + offs).to(tl.float32)

    inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    out = x * inv_rms * w

    tl.store(O_ptr + pid * so_m + offs * so_d, out.to(O_ptr.dtype.element_ty))


@torch.compiler.disable
def rms_norm(x, weight, eps=1e-6, out=None):
    """RMSNorm: ``out = weight * x / sqrt(mean(x**2, dim=-1, keepdim=True) + eps)``.

    x: (M, D) — D must be a power of 2 (the kernel loads the full row in one tile).
    weight: (D,)
    """
    assert x.dim() == 2
    M, D = x.shape
    assert weight.shape == (D,)
    assert x.stride(-1) == 1
    assert (D & (D - 1)) == 0, f"D={D} must be a power of 2"

    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape and out.stride(-1) == 1

    _rms_norm_kernel[(M,)](
        x, weight, out,
        eps,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        D=D,
    )
    return out

