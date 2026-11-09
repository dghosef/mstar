"""Resolve a policy a serving config names: a built-in's name, or ``"package.module:Class"`` for a class of the
deployment's own."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TypeVar

T = TypeVar("T")


def resolve_policy(key: str, spec: str, builtins: Mapping[str, type[T]], base: type[T]) -> type[T]:
    """The class ``spec`` names for the config key ``key``. Raises ``ValueError`` when it names no built-in, can't be
    imported, or isn't a subclass of ``base``."""
    if not isinstance(spec, str):
        raise ValueError(f"{key} must be one of {sorted(builtins)} or 'package.module:Class', got {spec!r}")
    if spec in builtins:
        return builtins[spec]
    module_name, sep, class_name = spec.partition(":")
    if not sep or not module_name or not class_name:
        raise ValueError(f"{key} must be one of {sorted(builtins)} or 'package.module:Class', got {spec!r}")
    try:
        cls = getattr(import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"{key} {spec!r} can't be imported: {exc}") from exc
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise ValueError(f"{key} {spec!r} must name a subclass of {base.__name__}")
    return cls
