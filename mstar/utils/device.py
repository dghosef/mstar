
from __future__ import annotations

from functools import lru_cache

import torch

from mstar.engine.accelerator_graph_backend import (
    AcceleratorGraphBackend,
    get_accelerator_graph_backend,
)


class DeviceCapability:
    """Query one device's capabilities without initializing its hardware.

    Runtime operations use PyTorch's generic stream, event, and accelerator
    APIs directly. Kernel dispatch remains with the resource that owns it.
    """

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self._runtime = getattr(torch, self.device.type, None)
        self.graph_backend: AcceleratorGraphBackend | None
        try:
            self.graph_backend = get_accelerator_graph_backend(self.device)
        except ValueError:
            self.graph_backend = None

    @staticmethod
    def of(device: torch.device | str) -> "DeviceCapability":
        return _device_capability(torch.device(device))

    @property
    def is_accelerator(self) -> bool:
        return self.device.type != "cpu" and all(
            hasattr(self._runtime, name)
            for name in ("Stream", "Event", "stream", "is_available")
        )

    @property
    def is_available(self) -> bool:
        return self.is_accelerator and bool(self._runtime.is_available())

    @property
    def pin_memory(self) -> bool:
        return self.is_available

    def is_capturing(self) -> bool:
        return (
            self.graph_backend is not None
            and self.is_available
            and self.graph_backend.is_current_stream_capturing()
        )


def to_device_nonblocking(t: torch.Tensor, device) -> torch.Tensor:
    """A host tensor onto ``device`` without blocking the host. A copy from
    pageable memory waits for everything already queued on the stream; a
    pinned non-blocking one does not."""
    device = torch.device(device)
    if t.device.type != "cpu" or not DeviceCapability.of(device).pin_memory:
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


@lru_cache(maxsize=None)
def _device_capability(device: torch.device) -> DeviceCapability:
    return DeviceCapability(device)


def to_device_nonblocking(t: torch.Tensor, device) -> torch.Tensor:
    """``t`` on ``device`` without a stream sync: a host tensor is pinned, then
    copied asynchronously. A tensor already on ``device`` comes back as is."""
    if t.device == torch.device(device):
        return t
    if t.device.type == "cpu" and not t.is_pinned():
        t = t.pin_memory()
    return t.to(device, non_blocking=True)
