"""Fused elementwise kernels for the Qwen-Image DiTs (Edit-2511/2509, 2.1), bit-exact with the eager modules they
replace.

Each kernel rounds to bf16 exactly where the eager ops do, so the fused and the eager forward give the same bits
(the served output stays identical to the reference pipeline's). Off CUDA, or for a shape a kernel does not take,
the callers run the eager ops (``fits``).

* :func:`qk_norm_rope` (one part) and :func:`joint_qk_norm_rope` (text and image parts into one joint tensor):
  ``QwenRMSNorm`` of each 128-wide head, then the complex rotary product. The variance is
  summed in the order of PyTorch's reduction for a contiguous 128-wide row: 32 lanes of 4 consecutive values,
  ``((x0 + x1) + x2) + x3``, then a shuffle-down tree (offsets 16, 8, 4, 2, 1); the rotation is CUDA's complex
  multiply, ``re = fma(a, c, -(b d))``, ``im = fma(a, d, b c)``. Text and image parts go straight into the joint
  q (or k) tensor, so there is no concatenation either.
* :func:`modulate`: ``x * (1 + scale) + shift`` with two parameter sets along the tokens (``zero_cond_t``: the
  target's tokens take the step's, the references' timestep 0's), one pass instead of the per-token select.
* :func:`gated_add`: ``x + gate * y``, the same two parameter sets.

Two more fuse ``nn.LayerNorm`` (no affine) in front of the modulation, :func:`norm_modulate` and
:func:`gated_norm_modulate` (the residual, then its norm, in one pass). They are not bit-exact: Triton sums the row
statistics in its own order; the step's output stays within the bf16 noise of the reference.

Every launch turns Triton's fp fusion off: with it on, the compiler folds a bf16 round trip between a multiply and
an add into one fused multiply-add, which drops the rounding the eager ops make there.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

_ROWS = 16
_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _halves(s, n: tl.constexpr):
        # [R, 2n] -> [:, :n] + [:, n:], a shuffle-down step at offset n
        a, b = tl.split(tl.permute(tl.reshape(s, (s.shape[0], 2, n)), (0, 2, 1)))
        return a + b

    @triton.jit
    def _qk_norm_rope_kernel(x_ptr, w_ptr, f_ptr, o_ptr, n_rows, S, H, eps,
                             sx_r, sx_s, sx_h, sf_r, sf_s, so_r, so_s, so_h, ROWS: tl.constexpr):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        rmask = (rows < n_rows)[:, None]
        h = (rows % H).to(tl.int64)
        s = ((rows // H) % S).to(tl.int64)
        r = (rows // (H * S)).to(tl.int64)
        d = tl.arange(0, 128)
        x = tl.load(x_ptr + (r * sx_r + s * sx_s + h * sx_h)[:, None] + d[None, :], mask=rmask, other=0.0)
        x = x.to(tl.float32)
        sq = x * x
        # lane t: elements 4t..4t+3, summed in order
        e0, e1 = tl.split(tl.reshape(sq, (ROWS, 32, 2, 2)))
        x0, x2 = tl.split(e0)
        x1, x3 = tl.split(e1)
        lane = ((x0 + x1) + x2) + x3
        lane = _halves(lane, 16)
        lane = _halves(lane, 8)
        lane = _halves(lane, 4)
        lane = _halves(lane, 2)
        lane = _halves(lane, 1)
        mean = tl.reshape(lane, (ROWS,)) * (1.0 / 128.0)
        rstd = tl.math.rsqrt(mean + eps)
        y = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        y = (tl.load(w_ptr + d).to(tl.float32)[None, :] * y).to(tl.bfloat16).to(tl.float32)
        a, b = tl.split(tl.reshape(y, (ROWS, 64, 2)))
        f_off = (r * sf_r + s * sf_s)[:, None] + 2 * tl.arange(0, 64)[None, :]
        c = tl.load(f_ptr + f_off, mask=rmask, other=0.0)
        sn = tl.load(f_ptr + f_off + 1, mask=rmask, other=0.0)
        re = tl.fma(a, c, -(b * sn))
        im = tl.fma(a, sn, b * c)
        out = tl.reshape(tl.join(re, im), (ROWS, 128)).to(tl.bfloat16)
        tl.store(o_ptr + (r * so_r + s * so_s + h * so_h)[:, None] + d[None, :], out, mask=rmask)

    @triton.jit
    def _params(p_ptr, rows_per_set, row, token, split, D, col, mask):
        # the parameter row of this token: set 0 (row) before ``split``, set 1 (rows_per_set + row) from it
        prow = tl.where(token < split, row, rows_per_set + row).to(tl.int64)
        return tl.load(p_ptr + prow * D + col, mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def _modulate_kernel(x_ptr, scale1p_ptr, shift_ptr, o_ptr, L, D, rows_per_set, split, BLOCK: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)          # over rows * L tokens
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        row, token = tok // L, tok % L
        x = tl.load(x_ptr + tok * D + col, mask=mask, other=0.0).to(tl.float32)
        scale1p = _params(scale1p_ptr, rows_per_set, row, token, split, D, col, mask)
        shift = _params(shift_ptr, rows_per_set, row, token, split, D, col, mask)
        m = (x * scale1p).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + tok * D + col, (m + shift).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _gated_add_kernel(x_ptr, gate_ptr, y_ptr, o_ptr, L, D, rows_per_set, split, BLOCK: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        row, token = tok // L, tok % L
        x = tl.load(x_ptr + tok * D + col, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + tok * D + col, mask=mask, other=0.0).to(tl.float32)
        gate = _params(gate_ptr, rows_per_set, row, token, split, D, col, mask)
        gy = (gate * y).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + tok * D + col, (x + gy).to(tl.bfloat16), mask=mask)


    @triton.jit
    def _ln_mod(r, s1p_ptr, sh_ptr, o_ptr, base, prow_off, cols, mask, D, eps):
        # LayerNorm (no affine) of the fp32 row r, rounded to bf16 as nn.LayerNorm's output, then modulated
        mean = tl.sum(r, axis=0) / D
        rc = tl.where(mask, r - mean, 0.0)
        rstd = tl.math.rsqrt(tl.sum(rc * rc, axis=0) / D + eps)
        y = (rc * rstd).to(tl.bfloat16).to(tl.float32)
        s1p = tl.load(s1p_ptr + prow_off + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(sh_ptr + prow_off + cols, mask=mask, other=0.0).to(tl.float32)
        m = (y * s1p).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + base + cols, (m + sh).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _norm_modulate_kernel(x_ptr, s1p_ptr, sh_ptr, o_ptr, L, D, rows_per_set, split, eps, BLOCK: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        row, token = tok // L, tok % L
        prow = tl.where(token < split, row, rows_per_set + row).to(tl.int64)
        x = tl.load(x_ptr + tok * D + cols, mask=mask, other=0.0).to(tl.float32)
        _ln_mod(x, s1p_ptr, sh_ptr, o_ptr, tok * D, prow * D, cols, mask, D, eps)

    @triton.jit
    def _gated_norm_modulate_kernel(x_ptr, g_ptr, y_ptr, s1p_ptr, sh_ptr, xo_ptr, o_ptr, L, D, rows_per_set, split,
                                    eps, BLOCK: tl.constexpr):
        tok = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        row, token = tok // L, tok % L
        prow = tl.where(token < split, row, rows_per_set + row).to(tl.int64)
        x = tl.load(x_ptr + tok * D + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + tok * D + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + prow * D + cols, mask=mask, other=0.0).to(tl.float32)
        r = (x + (g * y).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(xo_ptr + tok * D + cols, r, mask=mask)
        _ln_mod(r.to(tl.float32), s1p_ptr, sh_ptr, o_ptr, tok * D, prow * D, cols, mask, D, eps)


def fits(*tensors: torch.Tensor) -> bool:
    """Whether the fused kernels take these tensors: CUDA bf16 with Triton available."""
    return triton is not None and all(t.is_cuda and t.dtype == torch.bfloat16 for t in tensors)


def _norm_rope_into(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float, out: torch.Tensor) -> None:
    rows, seq, heads, _ = x.shape
    f = torch.view_as_real(freqs.contiguous())
    sf_r, sf_s = (0, f.stride(0)) if f.dim() == 3 else (f.stride(0), f.stride(1))
    n = rows * seq * heads
    _qk_norm_rope_kernel[(triton.cdiv(n, _ROWS),)](
        x, weight, f, out, n, seq, heads, eps, x.stride(0), x.stride(1), x.stride(2), sf_r, sf_s,
        out.stride(0), out.stride(1), out.stride(2), ROWS=_ROWS, enable_fp_fusion=False,
    )


@torch.library.custom_op("mstar::qwen_image_qk_norm_rope", mutates_args=())
def qk_norm_rope(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    """``apply_rope(QwenRMSNorm(x), freqs)``: ``x [N, H, 128]`` with ``freqs`` complex64 ``[N, 64]``, or
    ``x [R, S, H, 128]`` with ``[S, 64]`` / ``[R, S, 64]``; bf16, last dim contiguous. Returns a new tensor."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # one token axis: a single row of N tokens
    _norm_rope_into(x[None] if x.dim() == 3 else x, weight, freqs, eps, out[None] if x.dim() == 3 else out)
    return out


@qk_norm_rope.register_fake
def _(x, weight, freqs, eps):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@torch.library.custom_op("mstar::qwen_image_joint_qk_norm_rope", mutates_args=())
def joint_qk_norm_rope(
    txt: torch.Tensor, img: torch.Tensor, txt_weight: torch.Tensor, img_weight: torch.Tensor,
    txt_freqs: torch.Tensor, img_freqs: torch.Tensor, eps: float,
) -> torch.Tensor:
    """``cat([rope(norm_txt(txt), txt_freqs), rope(norm_img(img), img_freqs)], dim=1)``: ``txt [R, Lt, H, 128]`` and
    ``img [R, Li, H, 128]`` bf16 (last dim contiguous), ``txt_freqs`` complex64 ``[R, Lt, 64]`` (or ``[Lt, 64]``),
    ``img_freqs`` ``[Li, 64]``; each part written straight into its slice of the joint ``[R, Lt + Li, H, 128]``."""
    rows, lt, heads, dim = txt.shape
    out = torch.empty((rows, lt + img.shape[1], heads, dim), dtype=img.dtype, device=img.device)
    _norm_rope_into(txt, txt_weight, txt_freqs, eps, out[:, :lt])
    _norm_rope_into(img, img_weight, img_freqs, eps, out[:, lt:])
    return out


@joint_qk_norm_rope.register_fake
def _(txt, img, txt_weight, img_weight, txt_freqs, img_freqs, eps):
    rows, lt, heads, dim = txt.shape
    return txt.new_empty((rows, lt + img.shape[1], heads, dim))


@torch.library.custom_op("mstar::qwen_image_modulate", mutates_args=())
def modulate(x: torch.Tensor, scale1p: torch.Tensor, shift: torch.Tensor, split: int) -> torch.Tensor:
    """``x * scale1p + shift`` for ``x [R, L, D]`` contiguous and ``[sets * R, D]`` parameters: a token before
    ``split`` takes set 0's row, from it set 1's (one set: ``split = L``)."""
    rows, length, dim = x.shape
    out = torch.empty_like(x)
    _modulate_kernel[(rows * length, triton.cdiv(dim, _BLOCK))](
        x, scale1p, shift, out, length, dim, rows, split, BLOCK=_BLOCK, enable_fp_fusion=False,
    )
    return out


@modulate.register_fake
def _(x, scale1p, shift, split):
    return torch.empty_like(x)


@torch.library.custom_op("mstar::qwen_image_gated_add", mutates_args=())
def gated_add(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor, split: int) -> torch.Tensor:
    """``x + gate * y`` (``x``, ``y`` ``[R, L, D]`` contiguous), ``gate`` by token as in :func:`modulate`."""
    rows, length, dim = x.shape
    out = torch.empty_like(x)
    _gated_add_kernel[(rows * length, triton.cdiv(dim, _BLOCK))](
        x, gate, y, out, length, dim, rows, split, BLOCK=_BLOCK, enable_fp_fusion=False,
    )
    return out


@gated_add.register_fake
def _(x, gate, y, split):
    return torch.empty_like(x)


@torch.library.custom_op("mstar::qwen_image_norm_modulate", mutates_args=())
def norm_modulate(x: torch.Tensor, scale1p: torch.Tensor, shift: torch.Tensor, split: int, eps: float) -> torch.Tensor:
    """``LayerNorm(x) * scale1p + shift`` (no affine; parameters by token as :func:`modulate`) in one pass. Not
    bit-exact: the row statistics sum in Triton's order, not ``nn.LayerNorm``'s."""
    rows, length, dim = x.shape
    out = torch.empty_like(x)
    _norm_modulate_kernel[(rows * length,)](
        x, scale1p, shift, out, length, dim, rows, split, eps, BLOCK=triton.next_power_of_2(dim),
        enable_fp_fusion=False,
    )
    return out


@norm_modulate.register_fake
def _(x, scale1p, shift, split, eps):
    return torch.empty_like(x)


@torch.library.custom_op("mstar::qwen_image_gated_norm_modulate", mutates_args=())
def gated_norm_modulate(
    x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor, scale1p: torch.Tensor, shift: torch.Tensor, split: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(r, norm_modulate(r))`` for the residual ``r = x + gate * y`` (the gate rounded as :func:`gated_add`),
    reading ``x`` and ``y`` once. The norm is :func:`norm_modulate`'s."""
    rows, length, dim = x.shape
    res, out = torch.empty_like(x), torch.empty_like(x)
    _gated_norm_modulate_kernel[(rows * length,)](
        x, gate, y, scale1p, shift, res, out, length, dim, rows, split, eps, BLOCK=triton.next_power_of_2(dim),
        enable_fp_fusion=False,
    )
    return res, out


@gated_norm_modulate.register_fake
def _(x, gate, y, scale1p, shift, split, eps):
    return torch.empty_like(x), torch.empty_like(x)
