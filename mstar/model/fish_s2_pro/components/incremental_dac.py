"""Incremental decode of the Fish S2-Pro codec: a chunk's new frames only, from per-stream state.

Every stage of ``FishDACDecoder`` is causal: the quantizer's post transformer
attends over the last ``post_window`` frames, the convolutions pad on the left
only and the transposed convolutions trim on the right. So a stream's audio
equals one decode of the whole stream when each chunk carries, from the chunks
before it, the transformer's last ``post_window - 1`` keys and values per
layer, each causal conv's last ``(kernel - 1) * dilation`` input samples, each
overlapping transposed conv's last input sample, and the frame count (RoPE
positions continue from it). Zeros are a fresh stream's state: the decoder's
own left zero padding.

``codec_state_blocks`` names that state as recurrent-state pool blocks; the
codec node gathers its rows' blocks before the decode and scatters the new
ones back after it.

After the transformer the decode runs time-major (``[B, T, C]``), every conv a
2-D conv on channels-last tensors and weights, so cuDNN's NHWC kernels need no
layout transposes around each conv; transposed convs are one GEMM each.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.recurrent import RecurrentBlockConfig
from mstar.model.fish_s2_pro.components.dac import (
    CausalConvNet,
    CausalTransConvNet,
    ConvNeXtBlock,
    DecoderBlock,
    FishDACDecoder,
    ResidualUnit,
    apply_rotary_emb,
)
from mstar.model.fish_s2_pro.components.dac_layers import Snake1d
from mstar.model.fish_s2_pro.config import FishDACConfig

FRAMES = "frames"


def _stateful_convs(decoder: FishDACDecoder) -> list[tuple[str, nn.Module]]:
    """``(block name, module)`` for every conv with left state, in execution order."""
    convs: list[tuple[str, nn.Module]] = []
    for i, (_, convnext) in enumerate(decoder.quantizer.upsample):
        convs.append((f"up{i}.dwconv", convnext.dwconv))
    model = decoder.decoder.model
    convs.append(("dec0", model[0]))
    for b, block in enumerate(model[1:-3], start=1):
        convs.append((f"dec{b}.upsample", block.block[1]))
        convs += [(f"dec{b}.res{r}", unit.block[1]) for r, unit in enumerate(block.block[2:])]
    convs.append(("final", model[-2]))
    return convs


def _state_width(module: nn.Module) -> int:
    """Input samples a module carries from one chunk to the next."""
    if isinstance(module, CausalTransConvNet):
        return (module.kernel_size - module.stride) // module.stride
    return module.padding


def codec_state_blocks(cfg: FishDACConfig) -> dict[str, RecurrentBlockConfig]:
    """Pool blocks for one stream's decoder state (fp32, the codec's dtype),
    from the codec config alone, so the model declares them before any weights
    load; ``IncrementalDACDecoder`` checks them against the loaded modules."""
    with torch.device("meta"):
        return IncrementalDACDecoder(FishDACDecoder(cfg)).state_blocks()


class IncrementalDACDecoder(nn.Module):
    """``FishDACDecoder`` over a chunk's frames, from and to its stream's state."""

    def __init__(self, decoder: FishDACDecoder):
        super().__init__()
        self.decoder = decoder
        post = decoder.quantizer.post_module
        attn = post.layers[0].attention
        self.heads, self.head_dim = attn.n_head, attn.head_dim
        self.window = post.window
        # keys a new frame reads besides its own
        self.cache = self.window - 1
        self.max_frames = decoder.config.max_frames
        self.convs = _stateful_convs(decoder)
        model = decoder.decoder.model
        if not (isinstance(model[-3], Snake1d) and isinstance(model[-1], nn.Tanh)):
            raise TypeError("incremental decode expects the decoder to end Snake -> conv -> tanh")
        for block in model[1:-3]:
            if not isinstance(block, DecoderBlock) or not all(isinstance(u, ResidualUnit) for u in block.block[2:]):
                raise TypeError("incremental decode expects DecoderBlocks of ResidualUnits")
        for _, module in self.convs:
            if isinstance(module, CausalTransConvNet):
                if module.kernel_size != 2 * module.stride:
                    raise TypeError("incremental decode assumes overlapping transposed convs of kernel 2 * stride")
            elif module.stride != 1:
                raise TypeError("incremental decode assumes stride-1 causal convs")
        for tconv, convnext in decoder.quantizer.upsample:
            if tconv.kernel_size != tconv.stride or not isinstance(convnext, ConvNeXtBlock):
                raise TypeError("incremental decode assumes non-overlapping upsampling transposed convs")
        if decoder.decoder.model[0].conv.weight.device.type != "meta":
            # Channels-last weights, in place: ``weight.unsqueeze(2)`` is then the
            # NHWC 2-D weight, and the reference modules still run on the same storage.
            with torch.no_grad():
                for _, module in self.convs:
                    if isinstance(module, CausalConvNet):
                        _to_channels_last(module.conv)

    def state_blocks(self) -> dict[str, RecurrentBlockConfig]:
        dtype = torch.float32
        blocks = {FRAMES: RecurrentBlockConfig(shape=(1,), dtype=torch.long)}
        for layer in range(len(self.decoder.quantizer.post_module.layers)):
            for kv in "kv":
                shape = (self.heads, self.cache, self.head_dim)
                blocks[f"{kv}{layer}"] = RecurrentBlockConfig(shape=shape, dtype=dtype)
        for name, module in self.convs:
            blocks[name] = RecurrentBlockConfig(shape=(_state_width(module), module.conv.in_channels), dtype=dtype)
        return blocks

    def _transformer(self, x, state, new, lengths):
        """``x [B, T, D]`` -> the post transformer's output, attending over the cached keys."""
        post = self.decoder.quantizer.post_module
        batch, width, _ = x.shape
        frames = state[FRAMES].view(batch, 1)
        qpos = frames + torch.arange(width, device=x.device)
        kpos = torch.cat([frames - self.cache + torch.arange(self.cache, device=x.device), qpos], 1)
        rel = qpos[:, :, None] - kpos[:, None, :]
        # the cached keys a stream has not produced yet sit at negative positions
        mask = ((kpos[:, None, :] >= 0) & (rel >= 0) & (rel < self.window))[:, None]
        freqs_cis = post.freqs_cis[qpos.clamp(max=self.max_frames - 1)]
        size = self.heads * self.head_dim
        h = x
        for i, layer in enumerate(post.layers):
            attn = layer.attention
            q, k, v = attn.wqkv(layer.attention_norm(h)).split([size, size, size], dim=-1)
            q = apply_rotary_emb(q.view(batch, width, self.heads, self.head_dim), freqs_cis).transpose(1, 2)
            k = apply_rotary_emb(k.view(batch, width, self.heads, self.head_dim), freqs_cis).transpose(1, 2)
            v = v.view(batch, width, self.heads, self.head_dim).transpose(1, 2)
            keys = torch.cat([state[f"k{i}"], k], 2)
            values = torch.cat([state[f"v{i}"], v], 2)
            y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
            h = h + layer.attention_layer_scale(attn.wo(y.transpose(1, 2).reshape(batch, width, size)))
            h = h + layer.ffn_layer_scale(layer.feed_forward(layer.ffn_norm(h)))
            new[f"k{i}"], new[f"v{i}"] = _tail(keys, lengths, self.cache, 2), _tail(values, lengths, self.cache, 2)
        return post.norm(h)

    def forward(
        self, codes: torch.Tensor, state: Mapping[str, torch.Tensor], lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``codes [B, 10, width]``, each row's state and its real frame count
        ``lengths [B]`` -> float audio ``[B, width * hop]`` and the state after
        each row's real frames.

        Rows are zero-padded on the right to the batch's width. The decoder is
        causal, so padding never reaches a row's real audio (the audio past it
        is junk the caller drops), and each state block is read from the row's
        own last real samples.
        """
        decoder = self.decoder
        new: dict[str, torch.Tensor] = {}
        convs = iter(self.convs)
        rate = [1]   # input samples per frame of the next stateful module

        def conv(x):
            """The next stateful conv over ``x [B, T, C]`` from its state, time-major."""
            name, module = next(convs)
            keep = state[name].shape[1]
            xs = torch.cat([state[name], x], 1)
            new[name] = _tail(xs, lengths * rate[0], keep, 1)
            c = module.conv
            if isinstance(module, CausalTransConvNet):
                # Kernel 2 * stride: input i writes outputs i*stride .. i*stride + 2*stride - 1, so
                # each output block is its input's first half plus the previous input's second.
                stride = module.stride
                rate[0] *= stride
                y = _tconv_gemm(xs, c).unflatten(2, (2, stride))       # [B, 1 + T, 2, stride, C]
                y = y[:, 1:, 0] + y[:, :-1, 1]
                return y.flatten(1, 2) + c.bias
            x4 = xs.transpose(1, 2).unsqueeze(2)   # [B, C, 1, T], a channels-last view
            y = F.conv2d(x4, c.weight.unsqueeze(2), c.bias, dilation=(1, c.dilation[0]), groups=c.groups)
            return y.squeeze(2).transpose(1, 2)

        h = self._transformer(decoder.quantizer.embed(codes).transpose(1, 2), state, new, lengths)
        for tconv, convnext in decoder.quantizer.upsample:
            c = tconv.conv   # kernel == stride: each input writes its own output block
            h = _tconv_gemm(h, c).flatten(1, 2) + c.bias
            rate[0] *= tconv.stride
            y = convnext.pwconv2(convnext.act(convnext.pwconv1(convnext.norm(conv(h)))))
            h = h + convnext.gamma * y
        model = decoder.decoder.model
        h = conv(h)
        for block in model[1:-3]:
            h = conv(_snake(block.block[0], h))
            for unit in block.block[2:]:
                snake1, _, snake2, pointwise = unit.block
                c2 = pointwise.conv
                h = h + F.linear(_snake(snake2, conv(_snake(snake1, h))), c2.weight.squeeze(-1), c2.bias)
        wav = torch.tanh(conv(_snake(model[-3], h)))
        new[FRAMES] = state[FRAMES] + lengths.view(-1, 1)
        return wav.squeeze(-1), new


def _snake(module: Snake1d, x: torch.Tensor) -> torch.Tensor:
    """``Snake1d`` over a time-major ``x [B, T, C]``, in fp32 as its kernel computes."""
    dtype = torch.promote_types(x.dtype, torch.float32)
    alpha = module.alpha.reshape(-1).to(dtype)
    x32 = x.to(dtype)
    return (x32 + (alpha + 1e-9).reciprocal() * torch.sin(x32 * alpha).pow(2)).to(x.dtype)


def _tconv_gemm(x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
    """Every input sample's transposed-conv output, one GEMM: ``x [B, T, C_in]`` ->
    ``[B, T, kernel, C_out]`` (no bias, no overlap-add)."""
    weight = conv.weight   # [C_in, C_out, kernel]
    y = x @ weight.permute(0, 2, 1).reshape(weight.shape[0], -1)
    return y.unflatten(2, (weight.shape[2], weight.shape[1]))


def _to_channels_last(conv: nn.Module) -> None:
    w = conv.weight
    w.data = w.data.unsqueeze(2).contiguous(memory_format=torch.channels_last).squeeze(2)


def _tail(xs: torch.Tensor, valid: torch.Tensor, keep: int, dim: int) -> torch.Tensor:
    """Per row, the ``keep`` entries of ``xs`` along ``dim`` that end ``valid[row]``
    entries past its first ``keep`` (``xs`` is ``[keep of state | new]``)."""
    shape = [1] * xs.dim()
    shape[0], shape[dim] = -1, keep
    index = valid.view(-1, *[1] * (xs.dim() - 1)) + torch.arange(keep, device=xs.device).view(shape)
    size = list(xs.shape)
    size[dim] = keep
    return xs.gather(dim, index.expand(size))
