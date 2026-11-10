"""The serving config's LoRA adapters, resolved once before the conductor starts.

The API server reads every adapter its ``resources: lora:`` block names, refusing the
deployment if any cannot be served, and writes what it found back into the block: each
adapter's digest, which every later read must match, and the targets the adapters touch.
The conductor is handed that config, and spawns the workers with it, so every process and
rank builds the same pools and serves the weights the API server keyed the prefix cache
under.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

from mstar.engine.resources.lora.checkpoint import _SIDECAR_CONFIGS
from mstar.engine.resources.lora.config import ADAPTER_WEIGHTS_FILE, LORA_RESOURCE_KEY, LoraSpec, resolve_adapter_path

logger = logging.getLogger(__name__)


def lora_spec(specs: Iterable[Any]) -> LoraSpec | None:
    """The deployment's LoRA spec among its declared resource specs
    (``declared_resources``: whole, not yet split into a pool per node)."""
    return next((s for s in specs if isinstance(s, LoraSpec)), None)


def served_lora_targets(spec: LoraSpec) -> list[str] | None:
    """The targets a deployment keeps when its adapters are fixed at startup (runtime
    updates off, and no YAML ``targets``, which the caller checks): those one of its
    adapters touches, as the ``targets`` list every process then applies.
    Every other target would hold zero weights in every slot, yet cost its kernels on
    every step and its memory (the LM head's and the embedding's are vocabulary-sized),
    and a node left with none gets no pool. None when nothing narrows."""
    from mstar.engine.resources.lora.store import adapter_targets

    if spec.runtime_updates or not spec.adapters:
        return None
    used = set().union(*(adapter_targets(spec, path) for path in spec.adapters.values()))
    return None if not used or used == set(spec.targets) else sorted(used)


def refuse_unresolved_lora_adapters(model_config: Mapping[str, Any]) -> None:
    """Refuse a serving config whose ``lora:`` block names adapters with no pinned digest:
    workers spawned with it would serve whatever each of their reads finds, rather than
    the weights the API server keyed the prefix cache under."""
    block = (model_config.get("resources") or {}).get(LORA_RESOURCE_KEY)
    if not isinstance(block, Mapping):
        return  # the `resources:` override step reports a block that is not a mapping
    unpinned = sorted(set(block.get("adapters") or ()) - set(block.get("adapter_digests") or ()))
    if unpinned:
        raise ValueError(
            f"LoRA adapters {unpinned} have no pinned digest: the API server resolves the "
            "serving config's adapters (resolve_lora_adapters) and hands the conductor the result"
        )


def resolve_lora_adapters(spec: LoraSpec | None, model_config: dict) -> dict[str, bytes]:
    """Check every adapter ``spec`` serves (``check_adapters``: a bad one fails the
    deployment, every bad one named), write their digests and the targets they touch
    into ``model_config``'s ``resources: lora:`` block and onto ``spec``, which then is
    what every worker builds, and return the digests."""
    if spec is None or not spec.adapters:
        return {}
    from mstar.engine.resources.lora.store import check_adapters

    # a repo id becomes its snapshot's file here, once, and the workers read that file
    fetched, errors = {}, []
    for name, source in spec.adapters.items():
        try:
            fetched[name] = fetch_adapter_path(source)
        except Exception as exc:  # noqa: BLE001 - each adapter that cannot be fetched is named
            errors.append(f"{name!r}: {exc}")
    if errors:
        raise ValueError("the serving config's LoRA adapters cannot be fetched: " + "; ".join(errors))
    spec.apply_yaml_overrides(adapters=fetched)
    digests = check_adapters(spec)
    resolved: dict[str, Any] = {
        "adapters": fetched, "adapter_digests": {name: d.hex() for name, d in digests.items()},
    }
    block = model_config["resources"][LORA_RESOURCE_KEY]
    # a YAML list is the deployment's own choice, which nothing narrows
    targets = None if "targets" in block else served_lora_targets(spec)
    if targets is not None:
        logger.info("LoRA: serving only the targets the adapters touch: %s", targets)
        resolved["targets"] = targets
    block.update(resolved)
    spec.apply_yaml_overrides(**resolved)
    return digests


def fetch_adapter_path(source: str) -> str:
    """The local safetensors file ``source`` names: a local path as ``resolve_adapter_path``
    reads it, else a Hugging Face repo id (as vLLM's ``lora_path`` may be), whose
    ``adapter_model.safetensors`` and the configs among ``_SIDECAR_CONFIGS`` it has (where
    its alpha may be: ``sidecar_default_alpha``) alone are fetched into the HF cache, at one
    commit, from the cache alone under ``HF_HUB_OFFLINE``. A config is taken as absent only
    when the hub says so, now or in the cache's record of an earlier answer; one that could
    not be fetched fails the read, as the weights would, so an adapter is never served with
    another alpha than its files give it. Every worker reads the file checked here. Called
    where the API server checks adapters (startup and runtime loads), never while a config
    is parsed.

    A repo with no safetensors adapter is a ``ValueError``. Any other failure is not, so a
    client is told only that the adapter could not be read (``describe_load_error``), as
    for a local path that cannot: the answer says nothing of whether a path exists."""
    if os.path.exists(source) or os.path.isabs(source) or source.startswith(("~", ".")):
        return resolve_adapter_path(os.path.expanduser(source))
    from huggingface_hub import _CACHED_NO_EXIST, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import (
        HFValidationError,
        LocalEntryNotFoundError,
        RemoteEntryNotFoundError,
    )
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(source)
    except HFValidationError:
        return source  # neither a path nor a repo id: the read says it is not there
    try:
        path = hf_hub_download(source, ADAPTER_WEIGHTS_FILE)
    except RemoteEntryNotFoundError as exc:
        raise ValueError(
            f"the Hugging Face repo {source!r} has no {ADAPTER_WEIGHTS_FILE}: only safetensors "
            "adapters are served"
        ) from exc
    except LocalEntryNotFoundError as exc:
        raise OSError(
            f"the Hugging Face repo {source!r} is not in the HF cache, and downloads are off "
            "or the hub cannot be reached"
        ) from exc
    commit = os.path.basename(os.path.dirname(path))
    for config in _SIDECAR_CONFIGS:
        try:
            hf_hub_download(source, config, revision=commit)
        except RemoteEntryNotFoundError:
            pass  # not in the repo: the weights' own alphas, or PEFT's default
        except LocalEntryNotFoundError as exc:
            # the hub could not be asked (offline, unreachable, an error answer)
            if try_to_load_from_cache(source, config, revision=commit) is not _CACHED_NO_EXIST:
                raise OSError(f"the Hugging Face repo {source!r}'s {config} could not be fetched") from exc
    return path
