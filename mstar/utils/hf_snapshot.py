"""Resolving a Hugging Face repo id to a local snapshot directory.

Not model-specific: every model that reads a pipeline snapshot's component
configs needs it, so it lives here rather than in one model's config module.
"""

from __future__ import annotations

from pathlib import Path

# Weights in every format but safetensors, and training state: what a fetch that reads
# only safetensors leaves out (``ignore_patterns``). A narrow ``allow_patterns`` would
# also leave out the small files an offline snapshot check expects.
NON_SAFETENSORS_WEIGHTS = (
    "*.bin", "*.pt", "*.pth", "*.msgpack", "*.h5", "*.ot", "*.onnx_data", "optimizer*",
)


def resolve_snapshot_dir(
    model_path_hf: str, cache_dir: str | None = None, allow_patterns: list[str] | None = None,
    ignore_patterns: tuple[str, ...] | list[str] | None = None,
) -> Path:
    """Local snapshot directory of a pipeline repo: a local path is used as-is,
    a hub id resolves through the HF cache (configs + weights of every
    component, offline when ``HF_HUB_OFFLINE`` is set).

    ``allow_patterns`` limits a download to the components the model serves, for
    repos that ship alternatives (a second transformer, a prompt enhancer) the
    deployment never loads; ``ignore_patterns`` leaves files out (e.g.
    ``NON_SAFETENSORS_WEIGHTS``)."""
    local = Path(model_path_hf)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        model_path_hf, cache_dir=cache_dir, allow_patterns=allow_patterns,
        ignore_patterns=None if ignore_patterns is None else list(ignore_patterns),
    ))
