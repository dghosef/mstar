"""Streaming checkpoint -> native module loading for Qwen-Image-2.1.

The transformer keeps the diffusers names; q/k/v stack into ``to_qkv`` and the SwiGLU's
``gate_layer`` / ``proj`` into ``gate_proj`` (sharded under TP by the parallel linears'
loaders), and ``to_out.0`` drops its index. The text encoder and the vision tower come from
the Qwen3-VL checkpoint in ``text_encoder/``; the VAE is diffusers' own module.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.diffusion.qwen3.encoder import Qwen3HiddenStateEncoder
from mstar.model.components.diffusion.qwen3.weight_loading import (
    make_text_encoder,
    qkv_rules,
    qwen3_vl_text_skip,
    remap_qwen3_vl_text_key,
)
from mstar.model.components.diffusion.weight_loading import (
    iter_diffusers_component,
    iter_transformers_component,
    load_native,
    materialize,
)
from mstar.model.components.qwen3_vit import VisionRotaryEmbedding
from mstar.model.components.qwen_vision import QwenVisionConfig, QwenVisionModel
from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.qwen_image21.components.transformer import QwenImage21Transformer
from mstar.model.qwen_image21.config import QwenImage21Config, QwenImage21TransformerConfig

TRANSFORMER_RULES = [
    StackedParamRule(".attn.to_qkv", ".attn.to_q", "q"),
    StackedParamRule(".attn.to_qkv", ".attn.to_k", "k"),
    StackedParamRule(".attn.to_qkv", ".attn.to_v", "v"),
    StackedParamRule(".img_mlp.gate_proj", ".img_mlp.gate_layer", 0),
    StackedParamRule(".img_mlp.gate_proj", ".img_mlp.proj", 1),
]
_VISION_PREFIX = "model.visual."


def remap_transformer_key(name: str) -> str:
    return name.replace(".attn.to_out.0.", ".attn.to_out.")


def load_transformer(
    module: QwenImage21Transformer, weights, what: str = "Qwen-Image-2.1 transformer",
) -> QwenImage21Transformer:
    load_native(module, weights, remap_transformer_key, what, stacked_params=TRANSFORMER_RULES)
    module.set_prefix_modulation()
    return module


def build_transformer(
    config: QwenImage21TransformerConfig, snapshot: Path, device, comm_group: CommGroup | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> QwenImage21Transformer:
    with torch.device("meta"):
        dit = QwenImage21Transformer(config, comm_group)
    materialize(dit, dtype, device)
    return load_transformer(dit, iter_diffusers_component(snapshot / "transformer", device))


def build_text_encoder(config: QwenImage21Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16
                       ) -> Qwen3HiddenStateEncoder:
    """Qwen3-VL-8B's language model, every layer; the final norm, the LM head and the vision tower
    stay on disk."""
    with torch.device("meta"):
        encoder = make_text_encoder(config.text)
    materialize(encoder, dtype, device)
    skip = qwen3_vl_text_skip(config.text)
    return load_native(
        encoder, iter_transformers_component(Path(snapshot) / "text_encoder", device, skip=skip),
        remap_qwen3_vl_text_key, "Qwen-Image-2.1 text encoder", stacked_params=qkv_rules(config.text), skip=skip,
    )


def build_vision_tower(snapshot: Path, device, dtype: torch.dtype = torch.bfloat16) -> QwenVisionModel:
    """Qwen3-VL-8B's vision tower with its deepstack mergers, from the text encoder's shards. Its rotary
    ``inv_freq`` is derived (fp32, as the reference keeps it), so completeness is over persistent state."""
    source = Path(snapshot) / "text_encoder"
    config = QwenVisionConfig.from_hf(source)
    with torch.device("meta"):
        tower = QwenVisionModel(config)
    materialize(tower, dtype, device)
    tower.rotary_pos_emb.register_buffer(
        "inv_freq", VisionRotaryEmbedding(config).inv_freq.to(device), persistent=False,
    )
    weights = iter_transformers_component(source, device, skip=lambda n: not n.startswith(_VISION_PREFIX))
    loaded = load_weights_into(tower, weights, name_remapper=lambda n: n[len(_VISION_PREFIX):])
    missing = sorted(set(tower.state_dict()) - loaded)
    if missing:
        raise RuntimeError(f"Qwen-Image-2.1 vision tower: {len(missing)} unloaded parameters {missing[:5]}")
    return tower.eval()


def build_vae(snapshot: Path, device, dtype: torch.dtype = torch.bfloat16):
    """diffusers' ``AutoencoderKLQwenImage21``, in the pipeline's dtype."""
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(str(Path(snapshot) / "vae"), torch_dtype=dtype)
    return vae.to(device).eval()
