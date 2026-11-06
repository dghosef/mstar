"""A CUDA tensor one process shares with another through CUDA IPC.

``SharedTensor.of`` names a tensor in terms any process can open, and ``open``
maps that memory here without copying: a zero-copy view of the owner's
allocation, so a write on either side lands in the same bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor


@dataclass(frozen=True)
class SharedTensor:
    cuda_share: tuple[Any, ...]
    size: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int
    dtype: str
    requires_grad: bool

    @classmethod
    def of(cls, tensor: torch.Tensor) -> SharedTensor:
        return cls(
            cuda_share=tensor.untyped_storage()._share_cuda_(), size=tensor.size(), stride=tensor.stride(),
            offset=tensor.storage_offset(), dtype=str(tensor.dtype), requires_grad=tensor.requires_grad,
        )

    def open(self) -> torch.Tensor:
        (
            _storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
            ref_counter_handle, ref_counter_offset, event_handle, event_sync_required,
        ) = self.cuda_share
        # Import into the receiving stream's device context. Importing on the
        # exporter's device permits peer copies but not local stream word ops.
        return rebuild_cuda_tensor(
            torch.Tensor, self.size, self.stride, self.offset, torch.UntypedStorage,
            getattr(torch, self.dtype.split(".")[-1]), torch.cuda.current_device(), storage_handle, storage_size_bytes,
            storage_offset_bytes, self.requires_grad, ref_counter_handle, ref_counter_offset, event_handle,
            event_sync_required,
        )
