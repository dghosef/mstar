"""Standalone Qwen3-TTS codec harness: per-batch timing, kernel mix and streaming parity.

Builds the serving ``CodecSubmodule`` (same loader, Snake fusion and split
decode as the server) and drives its decode directly, so codec changes can be
measured in seconds instead of a server restart:

    python -m benchmark.qwen3tts_codec.codec_bench timing  --out DIR
    python -m benchmark.qwen3tts_codec.codec_bench kernels --out DIR
    python -m benchmark.qwen3tts_codec.codec_bench parity  --codes codes.pt --out DIR

``timing`` and ``kernels`` compare the window decode older checkouts serve (``base``, run from such a checkout:
transformer over context + new frames, conv stack over the new frames plus its
receptive field) with a stateful incremental decode (``incr``: new frames only,
the transformer's sliding-window KV and every causal conv's left samples carried
per request, gathered from and scattered to a slot pool each step as a declared
recurrent-state resource would). ``parity`` streams real code sequences through
both and through one whole-sequence decode (the reference's non-streaming
output) and reports SNR between them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def load_codec(dtype: str = "float32"):
    from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

    model = Qwen3TTSModel(MODEL, codec_dtype=dtype)
    model._ensure_full_snapshot()
    return model, model._create_codec_submodule("cuda")


# ---------------------------------------------------------------------------
# Base: the server's window decode
# ---------------------------------------------------------------------------


def base_decode(codec, codes: torch.Tensor, contexts: list[int], frames: list[int]) -> list[torch.Tensor]:
    """The server's decode of a batch of windows ``codes [B, Q, width]`` (zero-padded),
    trimmed as ``CodecSubmodule.postprocess`` trims: float samples per row."""
    width = codes.shape[-1]
    conv_start = [c - min(c, codec.conv_context) for c in contexts]
    trim = codec._trim.get(width, 0)
    trim = min([trim] + [c - s for c, s in zip(contexts, conv_start, strict=True)])
    wav = codec._decode(codes, torch.tensor(conv_start, device=codes.device), trim)
    up = codec.total_upsample
    out = []
    for row, (c, f, s) in enumerate(zip(contexts, frames, conv_start, strict=True)):
        start, end = (c - s - trim) * up, (f - s - trim) * up
        out.append(wav[row, start:end].float() / 32767)
    return out


# ---------------------------------------------------------------------------
# Incremental: new frames only, causal state carried per request
# ---------------------------------------------------------------------------


def _conv_modules(decoder):
    """Every module of the decoder with left state, in execution order:
    (name, kind, module) with kind 'conv' (causal conv, state = its left pad
    of input samples) or 'tconv' (transposed conv with kernel 2*stride, state
    = its last input sample)."""
    out = [("pre_conv", "conv", decoder.pre_conv)]
    for i, (_, convnext) in enumerate(decoder.upsample):
        out.append((f"up{i}.dw", "conv", convnext.dwconv))
    out.append(("dec0", "conv", decoder.decoder[0]))
    for b, block in enumerate(decoder.decoder[1:-2], start=1):
        out.append((f"dec{b}.t", "tconv", block.block[1]))
        for r, unit in enumerate(block.block[2:]):
            out.append((f"dec{b}.r{r}", "conv", unit.conv1))
    out.append(("final", "conv", decoder.decoder[-1]))
    return out


class IncrementalCodec(torch.nn.Module):
    """Decode only a window's new frames, from per-request state.

    Exact against one whole-sequence decode: every module of the 12 Hz decoder
    is causal, so carrying each causal conv's left input samples, each
    transposed conv's last input and the transformer's sliding-window keys and
    values (with RoPE positions continuing from the frames already decoded)
    reproduces what the whole decode computes for these frames. A fresh
    request's state is zeros, which is the reference's own left zero padding.

    ``layout="nwc"`` keeps the conv stack's activations time-major (``[B, T, C]``,
    what the transformer emits and the ConvNeXt blocks' norm and linears want)
    and runs every conv as a channels-last 2-D conv, so cuDNN's NHWC kernels
    need no layout transposes.
    """

    def __init__(self, decoder, layout: str = "ncw"):
        super().__init__()
        self.dec = decoder
        self.nwc = layout == "nwc"
        t = decoder.pre_transformer
        cfg = t.config
        self.heads, self.head_dim = cfg.num_attention_heads, cfg.head_dim
        self.window = cfg.sliding_window
        self.cache = self.window - 1   # keys a new frame reads besides its own
        self.layers = list(t.layers)
        self.convs = _conv_modules(decoder)
        dtype = next(decoder.parameters()).dtype
        self.state_shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "frames": ((1,), torch.long),
        }
        for li in range(len(self.layers)):
            for kv in "kv":
                self.state_shapes[f"{kv}{li}"] = ((self.heads, self.cache, self.head_dim), dtype)
        for name, kind, mod in self.convs:
            conv = mod.conv
            width = mod.padding if kind == "conv" else 1
            shape = (width, conv.in_channels) if self.nwc and name != "pre_conv" else (conv.in_channels, width)
            self.state_shapes[name] = (shape, dtype)
        # channels-last 2-D weights, made once
        self._w4: dict[int, torch.Tensor] = {}
        if self.nwc:
            mods = [m.conv for _, _, m in self.convs] + [tc.conv for tc, _ in decoder.upsample]
            for conv in mods:
                self._w4[id(conv)] = conv.weight.detach().unsqueeze(2).contiguous(memory_format=torch.channels_last)
            self._pw: dict[int, torch.Tensor] = {}
            for block in decoder.decoder[1:-2]:
                for unit in block.block[2:]:
                    self._pw[id(unit.conv2.conv)] = unit.conv2.conv.weight.detach().squeeze(-1).contiguous()
            self._snake = {}
            for m in decoder.modules():
                if type(m).__name__ == "FusedSnakeBeta":
                    self._snake[id(m)] = (m.alpha, m.inv_beta)

    def state_bytes(self) -> int:
        return sum(math.prod(s) * torch.empty((), dtype=d).element_size() for s, d in self.state_shapes.values())

    # -- channels-first (the reference layout) --

    @staticmethod
    def _conv(x, mod, state):
        xs = torch.cat([state, x], -1)
        conv = mod.conv
        y = F.conv1d(xs, conv.weight, conv.bias, stride=conv.stride, dilation=conv.dilation, groups=conv.groups)
        return y, xs[..., -state.shape[-1]:]

    @staticmethod
    def _tconv(x, mod, state):
        xs = torch.cat([state, x], -1)
        conv = mod.conv
        s = conv.stride[0]
        y = F.conv_transpose1d(xs, conv.weight, conv.bias, stride=s)
        return y[..., s: s + x.shape[-1] * s], xs[..., -1:]

    # -- time-major: [B, T, C] tensors, channels-last convs --

    def _conv_nwc(self, x, conv, state=None, transposed=False):
        xs = x if state is None else torch.cat([state, x], 1)
        x4 = xs.transpose(1, 2).unsqueeze(2)          # [B, C, 1, T], channels-last strides: a view
        w = self._w4[id(conv)]
        if transposed:
            y = F.conv_transpose2d(x4, w, conv.bias, stride=(1, conv.stride[0]))
        else:
            y = F.conv2d(x4, w, conv.bias, stride=(1, conv.stride[0]), dilation=(1, conv.dilation[0]),
                         groups=conv.groups)
        y = y.squeeze(2).transpose(1, 2)               # [B, T, C]; contiguous when y is channels-last
        return y, (None if state is None else xs[:, -state.shape[1]:])

    def _snake_nwc(self, mod, x):
        alpha, inv_beta = self._snake[id(mod)]
        a, b = alpha.to(x.dtype), inv_beta.to(x.dtype)
        return x + b * torch.sin(x * a).pow(2)

    def _transformer(self, x, state, new):
        t = self.dec.pre_transformer
        B, W, _ = x.shape
        h = t.input_proj(x)
        frames = state["frames"].view(B, 1)
        ar = torch.arange(W, device=x.device)
        qpos = frames + ar                                           # [B, W]
        kpos = torch.cat([frames - self.cache + torch.arange(self.cache, device=x.device), qpos], 1)
        rel = qpos[:, :, None] - kpos[:, None, :]                    # [B, W, S+W]
        mask = (kpos[:, None, :] >= 0) & (rel >= 0) & (rel < self.window)
        cos, sin = t.rotary_emb(h, qpos)
        cos, sin = cos[:, None], sin[:, None]
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            res = h
            hn = layer.input_layernorm(h)
            q = attn.q_proj(hn).view(B, W, self.heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(hn).view(B, W, self.heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(hn).view(B, W, self.heads, self.head_dim).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            K = torch.cat([state[f"k{li}"], k], 2)
            V = torch.cat([state[f"v{li}"], v], 2)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask[:, None], scale=attn.scaling)
            o = attn.o_proj(o.transpose(1, 2).reshape(B, W, -1))
            h = res + layer.self_attn_layer_scale(o)
            h = h + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
            new[f"k{li}"], new[f"v{li}"] = K[:, :, -self.cache:], V[:, :, -self.cache:]
        return t.output_proj(t.norm(h))                              # [B, W, latent]

    def forward(self, codes: torch.Tensor, state: dict[str, torch.Tensor]):
        """``codes [B, Q, W]``, ``state`` per row -> (float wav ``[B, W*up]``, new state)."""
        dec = self.dec
        new: dict[str, torch.Tensor] = {}
        convs = iter(self.convs)

        def conv(x):
            name, kind, mod = next(convs)
            if self.nwc and name != "pre_conv":
                y, new[name] = self._conv_nwc(x, mod.conv, state[name], transposed=kind == "tconv")
                if kind == "tconv":
                    s = mod.conv.stride[0]
                    y = y[:, s: s + x.shape[1] * s]
                return y
            y, new[name] = (self._conv if kind == "conv" else self._tconv)(x, mod, state[name])
            return y

        h = dec.quantizer.decode(codes)
        h = conv(h)                                              # pre_conv
        h = self._transformer(h.transpose(1, 2), state, new)    # [B, W, latent]
        if self.nwc:
            for tconv, convnext in dec.upsample:
                h, _ = self._conv_nwc(h, tconv.conv, transposed=True)   # kernel == stride: no state
                y = conv(h)                                      # ConvNeXt dwconv
                y = convnext.pwconv2(convnext.act(convnext.pwconv1(convnext.norm(y))))
                h = h + convnext.gamma * y
            h = conv(h)                                          # decoder[0]
            for block in dec.decoder[1:-2]:
                h = conv(self._snake_nwc(block.block[0], h))     # Snake, transposed conv
                for unit in block.block[2:]:
                    y = conv(self._snake_nwc(unit.act1, h))
                    c2 = unit.conv2.conv
                    h = h + F.linear(self._snake_nwc(unit.act2, y), self._pw[id(c2)], c2.bias)
            wav = conv(self._snake_nwc(dec.decoder[-2], h)).squeeze(-1)
        else:
            h = h.permute(0, 2, 1)
            for tconv, convnext in dec.upsample:
                h = tconv.conv(h)                                # kernel == stride: no state
                res = h
                y = conv(h).permute(0, 2, 1)                     # ConvNeXt dwconv
                y = convnext.pwconv2(convnext.act(convnext.pwconv1(convnext.norm(y))))
                h = res + (convnext.gamma * y).permute(0, 2, 1)
            h = conv(h)                                          # decoder[0]
            for block in dec.decoder[1:-2]:
                h = block.block[0](h)                            # Snake
                h = conv(h)                                      # transposed conv
                for unit in block.block[2:]:
                    res = h
                    y = conv(unit.act1(h))
                    y = unit.conv2.conv(unit.act2(y))            # kernel 1
                    h = y + res
            h = dec.decoder[-2](h)
            wav = conv(h).squeeze(1)
        new["frames"] = state["frames"] + codes.shape[-1]
        return wav.clamp(-1, 1), new


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class ModelIncremental(torch.nn.Module):
    """The model's ``IncrementalDecoder`` behind the harness prototype's interface
    (every row full width), so ``incr_ncw`` times what the server runs."""

    def __init__(self, decoder):
        super().__init__()
        from mstar.model.qwen3_tts.components.incremental_codec import IncrementalDecoder

        self.dec = decoder
        self.impl = IncrementalDecoder(decoder)
        self.state_shapes = {
            name: (block.shape, block.dtype) for name, block in self.impl.state_blocks().items()
        }

    def state_bytes(self) -> int:
        return sum(math.prod(s) * torch.empty((), dtype=d).element_size() for s, d in self.state_shapes.values())

    def forward(self, codes, state):
        lengths = torch.full((codes.shape[0],), codes.shape[-1], device=codes.device, dtype=torch.long)
        return self.impl(codes, state, lengths)


def make_incremental(decoder, layout: str):
    """The model's decoder when this checkout has it (``ncw``), else the prototype."""
    if layout == "ncw" and importlib.util.find_spec("mstar.model.qwen3_tts.components.incremental_codec"):
        return ModelIncremental(decoder)
    return IncrementalCodec(decoder, layout)


def has_window_decode(codec) -> bool:
    """Whether this checkout's codec is the window decode (the ``base`` arm)."""
    return hasattr(codec, "conv_context")


class SlotPool:
    """The harness's stand-in for a recurrent-state pool: one tensor per state
    block, ``[slots, *shape]``; a step gathers its rows and scatters them back."""

    def __init__(self, incr: IncrementalCodec, slots: int, device):
        self.blocks = {
            name: torch.zeros((slots, *shape), dtype=dtype, device=device)
            for name, (shape, dtype) in incr.state_shapes.items()
        }

    def gather(self, idx):
        return {name: blk.index_select(0, idx) for name, blk in self.blocks.items()}

    def scatter_(self, idx, new):
        for name, blk in self.blocks.items():
            blk.index_copy_(0, idx, new[name])

    def reset(self):
        for blk in self.blocks.values():
            blk.zero_()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _graph(fn, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def _time(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def base_step_fn(codec, B, width, context, compiled):
    q = codec.config.codec.num_quantizers
    codes = torch.randint(0, 2048, (B, q, width), device="cuda")
    conv_start = torch.full((B,), context - min(context, codec.conv_context), device="cuda", dtype=torch.long)
    trim = codec._trim.get(width, 0) if codec.split_decode else 0
    dec = torch.compile(codec._decode, dynamic=False) if compiled else codec._decode
    return lambda: dec(codes, conv_start, trim)


def incr_step_fn(incr, pool, B, width, compiled):
    q = incr.dec.config.num_quantizers
    codes = torch.randint(0, 2048, (B, q, width), device="cuda")
    idx = torch.arange(1, B + 1, device="cuda")

    def step():
        wav, new = incr(codes, pool.gather(idx))
        pool.scatter_(idx, new)
        return wav

    return torch.compile(step, dynamic=False) if compiled else step


def cmd_timing(args):
    torch.backends.cudnn.benchmark = True
    model, codec = load_codec(args.dtype)
    cfg = codec.config.codec
    incrs = {lay: make_incremental(codec.decoder, lay) for lay in args.layouts}
    pools = {lay: SlotPool(m, max(args.batches) + 1, "cuda") for lay, m in incrs.items()}
    print(f"incremental state: {next(iter(incrs.values())).state_bytes() / 2**20:.2f} MiB per request")
    rows = []
    # base windows: the ramp's (context, new) pairs as the stream policy makes them
    base = has_window_decode(codec)
    pairs = sorted(cfg.codec_window_contexts((0,))) if base else [(w, 0) for w in sorted(
        {*cfg.chunk_schedule, cfg.chunk_frames})]
    for B in args.batches:
        for window, context in pairs:
            new = window - context
            bucket = codec._bucket(window)
            for compiled in (True,) if args.compiled_only else (False, True):
                if base:
                    fn = base_step_fn(codec, B, bucket, context, compiled)
                    ms = _time(_graph(fn), args.iters)
                    rows.append(dict(arm="base", compiled=compiled, B=B, new=new, context=context, width=bucket,
                                     ms=ms))
                    print(json.dumps(rows[-1]), flush=True)
                for lay, incr in incrs.items():
                    fn = incr_step_fn(incr, pools[lay], B, new, compiled)
                    ms = _time(_graph(fn), args.iters)
                    rows.append(dict(arm=f"incr_{lay}", compiled=compiled, B=B, new=new, context=0, width=new, ms=ms))
                    print(json.dumps(rows[-1]), flush=True)
    for B in args.batches:   # fixed small chunks (--extra-widths frames, default 10 and 12)
        for new in args.extra_widths:
            for lay, incr in incrs.items():
                fn = incr_step_fn(incr, pools[lay], B, new, True)
                rows.append(dict(arm=f"incr_{lay}", compiled=True, B=B, new=new, context=0, width=new,
                                 ms=_time(_graph(fn), args.iters)))
                print(json.dumps(rows[-1]), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"timing_{args.dtype}{'_tf32' if args.tf32 else ''}.jsonl"
    (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


# ---------------------------------------------------------------------------
# Kernel mix
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("layout", ("nchwToNhwc", "nhwcToNchw", "transpose", "Transpose")),
    ("conv", ("conv", "Conv", "implicit_gemm", "xmma", "cudnn", "sm90_", "dgrad", "wgrad")),
    ("gemm", ("gemm", "Gemm", "cutlass", "cublas", "matmul")),
    ("attention", ("flash", "fmha", "attention", "sdpa")),
    ("copy", ("copy", "Copy", "cat", "index", "gather", "scatter")),
    ("triton", ("triton_",)),
]


def _category(name: str) -> str:
    for cat, keys in CATEGORIES:
        if any(k in name for k in keys):
            return cat
    return "elementwise/other"


def cmd_kernels(args):
    from torch.profiler import ProfilerActivity, profile

    torch.backends.cudnn.benchmark = True
    model, codec = load_codec(args.dtype)
    incrs = {lay: make_incremental(codec.decoder, lay) for lay in args.layouts}
    pools = {lay: SlotPool(m, args.batch + 1, "cuda") for lay, m in incrs.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = codec.config.codec
    cases = {}
    if has_window_decode(codec):
        full_ctx = cfg.left_context_frames
        width = codec._bucket(full_ctx + cfg.chunk_frames)
        cases["base_eager"] = base_step_fn(codec, args.batch, width, full_ctx, False)
        cases["base_compiled"] = base_step_fn(codec, args.batch, width, full_ctx, True)
    for lay, incr in incrs.items():
        cases[f"incr_{lay}_eager"] = incr_step_fn(incr, pools[lay], args.batch, cfg.chunk_frames, False)
        cases[f"incr_{lay}_compiled"] = incr_step_fn(incr, pools[lay], args.batch, cfg.chunk_frames, True)
    report = {}
    for label, fn in cases.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        by_cat, by_name, count = defaultdict(float), defaultdict(float), 0
        for evt in prof.events():
            if evt.device_type.name != "CUDA":
                continue
            us = evt.device_time_total if hasattr(evt, "device_time_total") else evt.cuda_time_total
            by_cat[_category(evt.name)] += us / args.iters
            by_name[evt.name[:120]] += us / args.iters
            count += 1
        total = sum(by_cat.values())
        report[label] = {
            "kernels_per_step": count / args.iters,
            "gpu_us_per_step": total,
            "by_category_pct": {k: round(100 * v / total, 1) for k, v in sorted(by_cat.items(), key=lambda x: -x[1])},
            "top": sorted(((round(v, 1), k) for k, v in by_name.items()), reverse=True)[:25],
        }
        keys = ("kernels_per_step", "gpu_us_per_step", "by_category_pct")
        print(label, json.dumps({k: report[label][k] for k in keys}))
        prof.export_chrome_trace(str(out / f"trace_{label}_{args.dtype}_b{args.batch}.json"))
    (out / f"kernels_{args.dtype}_b{args.batch}.json").write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def stream_windows(total: int, schedule, chunk: int, left: int):
    """(context, new) per window of ``ScheduledLeftContextChunkPolicy`` over a
    stream of ``total`` frames, the producer done at the end (a short final flush)."""
    out, delivered, i = [], 0, 0
    while delivered < total:
        size = schedule[i] if i < len(schedule) else chunk
        size = min(size, total - delivered)
        out.append((min(left, delivered), size))
        delivered += size
        i += 1
    return out


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    n = min(ref.numel(), test.numel())
    ref, test = ref[:n].double(), test[:n].double()
    noise = (ref - test).pow(2).sum().clamp_min(1e-20)
    return float(10 * torch.log10(ref.pow(2).sum() / noise))


@torch.no_grad()
def cmd_parity(args):
    """SNR of the streamed decodes against a float32 whole-sequence decode (the
    reference's non-streaming output), every arm int16-quantized like the server's."""
    _, ref_codec = load_codec("float32")
    _, codec = load_codec(args.dtype) if args.dtype != "float32" else (None, ref_codec)
    cfg = codec.config.codec
    incrs = {lay: make_incremental(codec.decoder, lay) for lay in args.layouts}
    pools = {lay: SlotPool(m, 2, "cuda") for lay, m in incrs.items()}
    idx = torch.tensor([1], device="cuda")
    up = codec.total_upsample
    q = lambda x: (x.float().clamp(-1, 1) * 32767).round().clamp(-32768, 32767) / 32767  # noqa: E731
    seqs = torch.load(args.codes)   # list of [T, Q] long tensors (EOS stripped)
    results = []
    for n, codes in enumerate(seqs[: args.limit]):
        codes = codes.cuda().long()
        T = codes.shape[0]
        full = q(ref_codec.decoder(codes.t()[None]).reshape(-1))
        full_dtype = q(codec.decoder(codes.t()[None]).reshape(-1))
        base, inc, off = [], {lay: [] for lay in incrs}, 0
        for pool in pools.values():
            pool.reset()
        windowed = has_window_decode(codec)
        schedule = stream_windows(T, cfg.chunk_schedule, cfg.chunk_frames, cfg.left_context_frames if windowed else 0)
        for context, new in schedule:
            if windowed:
                win = codes[off - context: off + new].t()[None]
                width = codec._bucket(context + new)
                base.append(base_decode(codec, F.pad(win, (0, width - win.shape[-1])), [context], [context + new])[0])
            for lay, incr in incrs.items():
                wav, state = incr(codes[off: off + new].t()[None], pools[lay].gather(idx))
                pools[lay].scatter_(idx, state)
                inc[lay].append(q(wav.reshape(-1)))
            off += new
        base = torch.cat(base) if windowed else full
        row = dict(item=n, frames=T, whole_dtype_vs_full=snr_db(full, full_dtype), base_vs_full=snr_db(full, base))
        for lay in incrs:
            wav = torch.cat(inc[lay])
            assert base.numel() == wav.numel() == T * up, (base.numel(), wav.numel(), T * up)
            row[f"incr_{lay}_vs_full"] = snr_db(full, wav)
            row[f"incr_{lay}_vs_base"] = snr_db(base, wav)
        results.append(row)
        print(json.dumps(row), flush=True)
    keys = [k for k in results[0] if k not in ("item", "frames")]
    summary = {k: dict(mean=sum(r[k] for r in results) / len(results), min=min(r[k] for r in results)) for k in keys}
    print("summary", json.dumps(summary))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"parity_{args.dtype}.json").write_text(json.dumps(dict(rows=results, summary=summary), indent=1))


@torch.no_grad()
def cmd_encode(args):
    """WAVs (24 kHz) -> list of ``[T, 16]`` code tensors via the speech tokenizer's encoder."""
    import soundfile
    from transformers import MimiConfig

    from mstar.model.loader import load_hf_weights
    from mstar.model.loader.iterators import iter_safetensors_shards
    from mstar.model.qwen3_tts.qwen3_tts_model import _load_buffers, _load_qwen3_tts_codec_classes

    model, _ = load_codec("float32")
    _, _, encoder_cls = _load_qwen3_tts_codec_classes()
    enc = encoder_cls(MimiConfig(**model.config.codec.encoder_config)).to("cuda")
    codec_dir = Path(model.local_dir) / "speech_tokenizer"

    def weights():
        for name, tensor in iter_safetensors_shards(codec_dir, device="cuda", prefix="encoder."):
            yield name.removeprefix("encoder."), tensor

    load_hf_weights(enc, weights())
    _load_buffers(enc, weights())
    enc.eval()
    seqs = []
    for path in sorted(Path(args.wav_dir).glob("*.wav"))[: args.limit]:
        wav, sr = soundfile.read(path, dtype="float32")
        if sr != 24000:
            raise ValueError(f"{path}: {sr} Hz, expected 24000")
        x = torch.from_numpy(wav).cuda().view(1, 1, -1)
        codes = enc.encode(input_values=x, return_dict=True).audio_codes[0, :16].t().contiguous().cpu()
        seqs.append(codes)
        print(path.name, tuple(codes.shape), flush=True)
    torch.save(seqs, args.out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("timing", "kernels", "parity"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--dtype", default="float32")
        s.add_argument("--iters", type=int, default=30)
    sub.choices["timing"].add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    sub.choices["timing"].add_argument("--extra-widths", type=int, nargs="+", default=[10, 12])
    sub.choices["timing"].add_argument("--compiled-only", action="store_true")
    sub.choices["kernels"].add_argument("--batch", type=int, default=16)
    sub.choices["parity"].add_argument("--codes", required=True)
    sub.choices["parity"].add_argument("--limit", type=int, default=20)
    e = sub.add_parser("encode")
    e.add_argument("--wav-dir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=40)
    for sp in sub.choices.values():
        sp.add_argument("--tf32", action="store_true", help="TF32 matmuls (cuDNN convs use TF32 by default)")
        if sp is not e:
            sp.add_argument("--layouts", nargs="+", default=["ncw", "nwc"])
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    with torch.inference_mode():
        {"timing": cmd_timing, "kernels": cmd_kernels, "parity": cmd_parity, "encode": cmd_encode}[args.cmd](args)


if __name__ == "__main__":
    main()
