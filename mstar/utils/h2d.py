"""Copy-only host-to-device staging, for work that runs beside a live CUDA graph.

Pre-planning builds step N+1's small tensors on the plan stream while step N's
graph replays. Any SM work it issues (a kernel, a ``fill_``, a D2D copy) slows
the graph; H2D copies and event waits do not. So host values go to the device
in ONE copy from pinned memory (``PinnedStager``), padding written on the host,
never via a device-side staging tensor. An ``H2DMirror`` skips re-copying
values a destination already holds, as in a steady decode batch.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np
import torch

from mstar.utils.device import DeviceCapability


class H2DMirror:
    """What the stager last copied into one device destination.

    Lets ``PinnedStager.copy_`` skip a copy whose values the destination
    already holds. Only sound for a destination nothing but that stager writes,
    and that lives as long as its mirror -- hence an object the owner keeps
    with it, not a cache keyed by address (the allocator reuses addresses).
    """

    __slots__ = ("_key", "_values")

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._values: np.ndarray | None = None

    def matches(self, key: tuple, values: np.ndarray) -> bool:
        return (
            self._key == key and self._values is not None
            and np.array_equal(self._values, values)
        )

    def record(self, key: tuple, values: np.ndarray) -> None:
        self._key = key
        self._values = values.copy()

    def invalidate(self) -> None:
        """The destination was written some other way."""
        self._key = self._values = None


class PinnedStager:
    """A persistent pinned arena for one dtype, used as a ring of segments.

    ``copy_`` writes values at the arena's cursor and issues one non-blocking
    H2D copy from there on the current stream. Events are per segment, not per
    copy: when the cursor leaves a segment it records one on each stream that
    copied out of it, and a segment is not rewritten until those have
    completed, so a copy still in flight never sees its source change. Most
    copies therefore make no event call at all. Thread-safe: the plan thread
    and the GPU thread may share one.

    On CUDA the copy is a direct ``cudaMemcpyAsync``: the torch ``copy_`` of
    two slices costs several times the copy itself in dispatch, and records
    a host-allocator event of its own. Elsewhere it is the torch copy.
    """

    # a segment holds many copies, so leaving one (an event per stream) is rare
    MIN_SEGMENT_BYTES = 32 << 10

    def __init__(self, dtype: torch.dtype, numel: int = 256, depth: int = 8):
        self.dtype = dtype
        self._np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        self._itemsize = self._np_dtype.itemsize
        self._depth = depth
        self._seg_numel = 0
        self._seg = 0  # the segment the cursor is in
        self._off = 0  # the cursor, within that segment
        # per segment: the streams it was copied out on since it was entered
        # (raw handles on the cudaMemcpyAsync path), then, once left, the
        # events recorded on them
        self._streams: list[list] = [[] for _ in range(depth)]
        self._events: list[list[torch.Event]] = [[] for _ in range(depth)]
        self._lock = threading.Lock()
        self._pinned = torch.accelerator.is_available()
        self._memcpy = _cuda_memcpy_async() if self._pinned else None
        self.copies = 0  # copies issued, for tests and accounting
        self._grow(max(numel, self.MIN_SEGMENT_BYTES // self._itemsize))

    def _grow(self, numel: int) -> None:
        # Nothing may still be copying out of the arena being replaced.
        self._leave_segment()
        for events in self._events:
            for ev in events:
                ev.synchronize()
            events.clear()
        self._seg_numel = max(numel, 2 * self._seg_numel, 1)
        self._arena = torch.empty(self._depth * self._seg_numel, dtype=self.dtype, pin_memory=self._pinned)
        self._view = self._arena.numpy()
        self._ptr = self._arena.data_ptr()
        self._seg = self._off = 0

    def _leave_segment(self) -> None:
        streams = self._streams[self._seg]
        self._events[self._seg] = [
            (torch.cuda.ExternalStream(s) if isinstance(s, int) else s).record_event() for s in streams
        ]
        streams.clear()

    def _next_segment(self) -> None:
        self._leave_segment()
        self._seg = (self._seg + 1) % self._depth
        self._off = 0
        for ev in self._events[self._seg]:
            ev.synchronize()  # the copies out of this segment are done
        self._events[self._seg].clear()

    def copy_(
        self,
        dst: torch.Tensor,
        values: Sequence | np.ndarray,
        pad_value=None,
        mirror: H2DMirror | None = None,
    ) -> None:
        """``dst[:len(values)] = values`` in one non-blocking H2D copy.

        With ``pad_value``, the rest of ``dst`` is set to it by the same copy
        (the padding is written on the host), so ``dst`` is fully overwritten
        without a device-side fill. ``dst`` must be contiguous.

        With ``mirror`` (``dst``'s, see ``H2DMirror``), nothing is issued when
        ``dst`` already holds exactly this: the copy that put it there is
        ahead of anything that reads ``dst`` after this call.
        """
        n = len(values)
        total = dst.numel() if pad_value is not None else n
        if n > total:
            raise ValueError(f"{n} values do not fit a destination of {total}")
        if total == 0:
            return
        if mirror is not None:
            values = np.asarray(values, dtype=self._np_dtype)
            key = (total, pad_value)
        direct = self._memcpy is not None and dst.is_cuda
        if direct:
            stream = torch._C._cuda_getCurrentRawStream(dst.get_device())
        elif self._pinned and DeviceCapability.of(dst.device).is_accelerator:
            stream = torch.accelerator.current_stream(dst.device)
        else:
            host = np.asarray(values)
            dst.view(-1)[:n].copy_(torch.as_tensor(host, dtype=self.dtype))
            if pad_value is not None:
                dst.view(-1)[n:].fill_(pad_value)
            return
        with self._lock:
            if mirror is not None and mirror.matches(key, values):
                return
            if total > self._seg_numel:
                self._grow(total)
            elif self._off + total > self._seg_numel:
                self._next_segment()
            at = self._seg * self._seg_numel + self._off
            self._off += total
            view = self._view
            view[at:at + n] = values
            if pad_value is not None:
                view[at + n:at + total] = pad_value
            if direct:
                err = self._memcpy(
                    dst.data_ptr(), self._ptr + at * self._itemsize, total * self._itemsize, _H2D, stream,
                )
                if err:
                    raise RuntimeError(f"cudaMemcpyAsync failed with error {err}")
            else:
                dst.view(-1)[:total].copy_(self._arena[at:at + total], non_blocking=True)
            streams = self._streams[self._seg]
            if stream not in streams:
                streams.append(stream)
            self.copies += 1
            if mirror is not None:
                mirror.record(key, values)


_H2D = 1  # cudaMemcpyHostToDevice
_MEMCPY = None


def _cuda_memcpy_async():
    """libcudart's ``cudaMemcpyAsync`` through ctypes, or None off CUDA (a
    ROCm or XPU build takes the torch copy). Loaded by the soname of torch's
    CUDA major, which reaches the copy torch already loaded. ``PyDLL``: the
    call keeps the GIL, which for a call of a few microseconds is cheaper
    than handing it to another thread and waiting to get it back."""
    global _MEMCPY
    if _MEMCPY is None:
        _MEMCPY = False
        if torch.version.cuda and torch.cuda.is_available():
            import ctypes

            try:
                lib = ctypes.PyDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
            except OSError:
                return None
            fn = lib.cudaMemcpyAsync
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
            fn.restype = ctypes.c_int
            _MEMCPY = fn
    return _MEMCPY or None


def to_device_nonblocking(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """``tensor`` on ``device`` without the host waiting: a CPU source goes
    through pinned memory, so the copy queues behind the work already on the
    stream instead of draining it first (a pageable copy blocks until then)."""
    device = torch.device(device)
    if device.type != "cuda" or tensor.device.type != "cpu":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)
