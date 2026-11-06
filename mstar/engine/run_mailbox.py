"""A carried signal's way across GPUs, for a run with a part on each.

A run's rounds carry each signal from the node that makes it to the node that
reads it (``RunPart``). On one GPU that is a copy between captured buffers, ordered
by the stream. Between two GPUs the reader's process owns a mailbox: ``data``, a
buffer the size of the signal, and ``word``, a 32-bit round counter, both on the
reader's GPU. The maker's process opens them once, at load, and each round copies
into ``data`` and writes ``word``; the reader's stream waits on ``word``, in its
own memory, then copies ``data`` into the captured input.

One mailbox per signal holds: a run is rounds in turn, so the maker's next write
follows the reader's answer to this one. Both sides count rounds on, across runs, so
a word is never reset.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mstar.utils.cuda_share import SharedTensor


@dataclass(frozen=True)
class MailboxShare:
    """What a mailbox's owner sends the process that will write it."""
    data: SharedTensor
    word: SharedTensor


@dataclass(frozen=True)
class RunMailbox:
    data: torch.Tensor
    word: torch.Tensor

    @classmethod
    def build(cls, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> RunMailbox:
        return cls(torch.zeros(shape, dtype=dtype, device=device), torch.zeros(1, dtype=torch.int32, device=device))

    def share(self) -> MailboxShare:
        return MailboxShare(SharedTensor.of(self.data), SharedTensor.of(self.word))

    @classmethod
    def open(cls, share: MailboxShare) -> RunMailbox:
        return cls(share.data.open(), share.word.open())

    @property
    def word_address(self) -> int:
        return self.word.data_ptr()


def mailbox_refusal(device: torch.device, peer: torch.device) -> str | None:
    """Why ``device`` can't write a mailbox on ``peer``, or None: it needs CUDA 12's
    stream word ops and peer access between the two."""
    if device.type != "cuda" or peer.type != "cuda":
        return f"no peer mapping between {device} and {peer}"
    if int((torch.version.cuda or "0").split(".")[0]) < 12:
        return f"CUDA {torch.version.cuda} has no stream word ops on every device (12 and up)"
    if not torch.cuda.can_device_access_peer(device.index, peer.index):
        return f"{device} has no peer access to {peer}"
    return None
