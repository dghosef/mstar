"""Incremental decode of the 12 Hz codec: a chunk's new frames only, from per-request state.

Every module of the speech-tokenizer decoder is causal: the convolutions pad on
the left only, the transposed convolutions trim on the right, and the
transformer attends over a sliding window of earlier frames. So a stream's audio
equals one whole-sequence decode when each chunk carries, from the chunks before
it, each causal conv's last ``padding`` input samples, each transposed conv's
last input sample, the transformer's last ``sliding_window - 1`` keys and
values per layer, and the frame count (RoPE positions continue from it).
Zeros are a fresh stream's state: the decoder's own left zero padding.

``CodecState`` names that state as recurrent-state pool blocks; the codec node
gathers its rows' blocks from the pool before the decode and scatters the new
ones back after it.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.recurrent import RecurrentBlockConfig

FRAMES = "frames"
# the reference decoder's conv geometry (fixed in its module definitions, not its config)
_KERNEL, _PRE_CONV_KERNEL, _DILATIONS = 7, 3, (1, 3, 9)


def codec_state_blocks(config, dtype: torch.dtype) -> dict[str, RecurrentBlockConfig]:
    """Pool blocks for one stream's decoder state, from ``Qwen3TTSCodecConfig``
    (declared before any weights load). ``IncrementalDecoder`` checks them
    against the loaded modules."""
    cache = config.sliding_window - 1
    blocks = {FRAMES: RecurrentBlockConfig(shape=(1,), dtype=torch.long)}
    for layer in range(config.num_hidden_layers):
        for kv in "kv":
            blocks[f"{kv}{layer}"] = RecurrentBlockConfig(
                shape=(config.num_attention_heads, cache, config.head_dim), dtype=dtype,
            )

    def conv(name, channels, width):
        blocks[name] = RecurrentBlockConfig(shape=(channels, width), dtype=dtype)

    conv("pre_conv", config.codebook_dim, _PRE_CONV_KERNEL - 1)
    for i in range(len(config.upsampling_ratios)):
        conv(f"up{i}.dwconv", config.latent_dim, _KERNEL - 1)
    conv("dec0", config.latent_dim, _KERNEL - 1)
    for b in range(1, len(config.upsample_rates) + 1):
        conv(f"dec{b}.upsample", config.decoder_dim // 2 ** (b - 1), 1)
        for r, dilation in enumerate(_DILATIONS):
            conv(f"dec{b}.res{r}", config.decoder_dim // 2 ** b, (_KERNEL - 1) * dilation)
    conv("final", config.decoder_dim // 2 ** len(config.upsample_rates), _KERNEL - 1)
    return blocks


def _stateful_convs(decoder: nn.Module) -> list[tuple[str, bool, nn.Module]]:
    """``(block name, transposed, module)`` for every module with left state, in execution order."""
    convs = [("pre_conv", False, decoder.pre_conv)]
    convs += [(f"up{i}.dwconv", False, convnext.dwconv) for i, (_, convnext) in enumerate(decoder.upsample)]
    convs.append(("dec0", False, decoder.decoder[0]))
    for b, block in enumerate(decoder.decoder[1:-2], start=1):
        convs.append((f"dec{b}.upsample", True, block.block[1]))
        convs += [(f"dec{b}.res{r}", False, unit.conv1) for r, unit in enumerate(block.block[2:])]
    convs.append(("final", False, decoder.decoder[-1]))
    return convs


class IncrementalDecoder(nn.Module):
    """The reference decoder's forward over a chunk's frames, from and to its state."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        transformer = decoder.pre_transformer
        config = transformer.config
        self.heads, self.head_dim = config.num_attention_heads, config.head_dim
        if config.num_key_value_heads != self.heads:
            raise ValueError("the 12 Hz codec transformer is expected to have as many KV heads as query heads")
        self.window = config.sliding_window
        # keys a new frame reads besides its own
        self.cache = self.window - 1
        self.layers = list(transformer.layers)
        self.convs = _stateful_convs(decoder)
        for _, transposed, module in self.convs:
            conv = module.conv
            if transposed and conv.kernel_size[0] != 2 * conv.stride[0]:
                raise TypeError("incremental decode assumes transposed convs of kernel 2 * stride")
            if not transposed and conv.stride[0] != 1:
                raise TypeError("incremental decode assumes stride-1 causal convs")
        for tconv, _ in decoder.upsample:
            if tconv.conv.kernel_size != tconv.conv.stride:
                raise TypeError("incremental decode assumes the upsampling transposed convs do not overlap")

    def state_blocks(self) -> dict[str, RecurrentBlockConfig]:
        """The blocks this decoder's modules read and write (``codec_state_blocks`` must match)."""
        dtype = next(self.decoder.parameters()).dtype
        blocks = {FRAMES: RecurrentBlockConfig(shape=(1,), dtype=torch.long)}
        for layer in range(len(self.layers)):
            for kv in "kv":
                blocks[f"{kv}{layer}"] = RecurrentBlockConfig(
                    shape=(self.heads, self.cache, self.head_dim), dtype=dtype,
                )
        for name, transposed, module in self.convs:
            width = 1 if transposed else module.padding
            blocks[name] = RecurrentBlockConfig(shape=(module.conv.in_channels, width), dtype=dtype)
        return blocks

    def _transformer(self, x, state, new, lengths):
        transformer = self.decoder.pre_transformer
        batch, width, _ = x.shape
        h = transformer.input_proj(x)
        frames = state[FRAMES].view(batch, 1)
        qpos = frames + torch.arange(width, device=x.device)
        kpos = torch.cat([frames - self.cache + torch.arange(self.cache, device=x.device), qpos], 1)
        rel = qpos[:, :, None] - kpos[:, None, :]
        # the cached keys a stream has not produced yet sit at negative positions
        mask = ((kpos[:, None, :] >= 0) & (rel >= 0) & (rel < self.window))[:, None]
        cos, sin = transformer.rotary_emb(h, qpos)
        cos, sin = cos[:, None], sin[:, None]
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            hn = layer.input_layernorm(h)
            q, k, v = (
                proj(hn).view(batch, width, self.heads, self.head_dim).transpose(1, 2)
                for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
            )
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            keys = torch.cat([state[f"k{i}"], k], 2)
            values = torch.cat([state[f"v{i}"], v], 2)
            o = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, scale=attn.scaling)
            h = h + layer.self_attn_layer_scale(attn.o_proj(o.transpose(1, 2).reshape(batch, width, -1)))
            h = h + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(h)))
            new[f"k{i}"], new[f"v{i}"] = _tail(keys, lengths, self.cache, 2), _tail(values, lengths, self.cache, 2)
        return transformer.output_proj(transformer.norm(h))

    def forward(
        self, codes: torch.Tensor, state: Mapping[str, torch.Tensor], lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``codes [B, quantizers, width]``, each row's state and its real frame
        count ``lengths [B]`` -> float audio ``[B, width * upsample]`` and the
        state after each row's real frames.

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
            name, transposed, module = next(convs)
            keep = state[name].shape[-1]
            xs = torch.cat([state[name], x], -1)
            new[name] = _tail(xs, lengths * rate[0], keep, 2)
            c = module.conv
            if transposed:
                stride = c.stride[0]
                rate[0] *= stride
                y = F.conv_transpose1d(xs, c.weight, c.bias, stride=stride)
                return y[..., stride: stride + x.shape[-1] * stride]
            return F.conv1d(xs, c.weight, c.bias, dilation=c.dilation, groups=c.groups)

        h = conv(decoder.quantizer.decode(codes))
        h = self._transformer(h.transpose(1, 2), state, new, lengths).transpose(1, 2)
        for tconv, convnext in decoder.upsample:
            h = tconv.conv(h)
            rate[0] *= tconv.conv.stride[0]
            y = conv(h).transpose(1, 2)
            y = convnext.pwconv2(convnext.act(convnext.pwconv1(convnext.norm(y))))
            h = h + (convnext.gamma * y).transpose(1, 2)
        h = conv(h)
        for block in decoder.decoder[1:-2]:
            h = conv(block.block[0](h))
            for unit in block.block[2:]:
                h = h + unit.conv2.conv(unit.act2(conv(unit.act1(h))))
        wav = conv(decoder.decoder[-2](h))
        new[FRAMES] = state[FRAMES] + lengths.view(-1, 1)
        return wav.clamp(-1, 1).squeeze(1), new


def _tail(xs: torch.Tensor, valid: torch.Tensor, keep: int, dim: int) -> torch.Tensor:
    """Per row, the ``keep`` entries of ``xs`` along ``dim`` that end ``valid[row]``
    entries past its first ``keep`` (``xs`` is ``[keep of state | new]``)."""
    shape = [1] * xs.dim()
    shape[0], shape[dim] = -1, keep
    index = (valid.view(-1, *[1] * (xs.dim() - 1)) + torch.arange(keep, device=xs.device).view(shape))
    size = list(xs.shape)
    size[dim] = keep
    return xs.gather(dim, index.expand(size))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
