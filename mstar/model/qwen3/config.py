"""Walks, resource keys and the reader of a text checkpoint's ``config.json``."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mstar.model.components.qwen3_lm import Qwen3LMConfig

LLM_NODE = "LLM"
PREFILL_WALK = "prefill"
DECODE_WALK = "decode"
DECODE_LOOP = "decode_loop"

KV_CACHE = "kv_cache"
ATTN = "attn"
ROPE = "rope"
SAMPLER = "sampler"


@dataclass(frozen=True)
class DecoderPart:
    """A piece of a decoder layer: the ``config.json`` keys it answers for, and
    a ``check`` refusing the values it doesn't implement."""

    keys: frozenset[str]
    check: Callable[[Any], None]


_ROPE_KEYS = {
    "default": {"rope_type", "rope_theta"},
    "llama3": {
        "rope_type", "rope_theta", "factor", "low_freq_factor", "high_freq_factor",
        "original_max_position_embeddings",
    },
}


def _check_full_attention(cfg) -> None:
    rope = cfg.rope_parameters
    if set(rope) - _ROPE_KEYS.get(rope.get("rope_type", "default"), set()):
        raise ValueError(f"rope_parameters {rope!r} (scaled RoPE other than llama3) is not supported")
    # the reference applies a window whenever this is set, and Mistral's default is 4096
    if getattr(cfg, "sliding_window", None) is not None:
        raise ValueError(f"sliding-window attention (sliding_window {cfg.sliding_window}) is not supported")
    if getattr(cfg, "attention_bias", False):
        raise ValueError("attention_bias is not supported")


def _check_gated_mlp(cfg) -> None:
    if cfg.hidden_act != "silu":
        raise ValueError(f"hidden_act {cfg.hidden_act!r} is not supported")
    if getattr(cfg, "mlp_bias", False):
        raise ValueError("mlp_bias is not supported")
    if getattr(cfg, "pretraining_tp", 1) != 1:
        raise ValueError(f"pretraining_tp {cfg.pretraining_tp} is not supported")


FULL_ATTENTION = DecoderPart(frozenset({
    "num_attention_heads", "num_key_value_heads", "head_dim", "rope_theta", "rope_scaling", "rope_parameters",
    "attention_bias", "sliding_window", "use_sliding_window", "max_window_layers", "layer_types",
    "max_position_embeddings",
}), _check_full_attention)
GATED_MLP = DecoderPart(frozenset({"intermediate_size", "hidden_act", "mlp_bias", "pretraining_tp"}), _check_gated_mlp)

# a layer kind (transformers' ``layer_types``) -> the parts it is built from
LAYER_KINDS = {"full_attention": (FULL_ATTENTION, GATED_MLP)}
_DECODER_KEYS = frozenset({"hidden_size", "num_hidden_layers", "vocab_size", "rms_norm_eps", "tie_word_embeddings"})
# keys that don't change what the model computes at inference
_IGNORED_KEYS = frozenset({
    "_name_or_path", "architectures", "attention_dropout", "bos_token_id", "dtype", "eos_token_id",
    "initializer_range", "model_type", "pad_token_id", "torch_dtype", "transformers_version", "unsloth_fixed",
    "use_cache",
})
# architecture -> (model_type, per-head QK norm)
_FAMILIES = {
    "Qwen3ForCausalLM": ("qwen3", True),
    "LlamaForCausalLM": ("llama", False),
    "MistralForCausalLM": ("mistral", False),
}


def qwen3_lm_config(hf: dict) -> Qwen3LMConfig:
    """The decoder a text checkpoint's ``config.json`` describes. Values come from
    transformers' own config class, so defaults and RoPE spelling are the
    reference's. A key no part of the decoder reads is refused: a config can
    change the output without changing a weight's shape."""
    from transformers import AutoConfig

    architecture = (hf.get("architectures") or [None])[0]
    model_type, qk_norm = _FAMILIES.get(architecture, (None, False))
    if model_type is None or model_type != hf.get("model_type"):
        raise ValueError(f"unsupported checkpoint: architecture {architecture!r}, model_type {hf.get('model_type')!r}")
    cfg = AutoConfig.for_model(model_type, **{k: v for k, v in hf.items() if k != "model_type"})
    kinds = set(getattr(cfg, "layer_types", None) or ["full_attention"])
    if kinds - LAYER_KINDS.keys():
        raise ValueError(f"layer kinds {sorted(kinds - LAYER_KINDS.keys())} are not supported")
    parts = {part for kind in kinds for part in LAYER_KINDS[kind]}
    unread = hf.keys() - _DECODER_KEYS - _IGNORED_KEYS - {key for part in parts for key in part.keys}
    if unread:
        raise ValueError(f"config.json keys nothing here reads: {sorted(unread)}")
    for part in parts:
        part.check(cfg)
    rope = cfg.rope_parameters
    scaling = {} if rope.get("rope_type") != "llama3" else {
        "rope_scale": rope["factor"],
        "rope_low_freq_factor": rope["low_freq_factor"],
        "rope_high_freq_factor": rope["high_freq_factor"],
        "rope_old_context_len": rope["original_max_position_embeddings"],
    }
    return Qwen3LMConfig.from_hf({**cfg.to_dict(), "rope_theta": rope["rope_theta"], "qk_norm": qk_norm, **scaling})
