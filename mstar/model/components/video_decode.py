"""Sampled-frame video decode, split across the clip's keyframe intervals.

Decoding a frame means decoding every frame from the keyframe before it, so a
sample spread over a whole clip costs a decode of most of the clip on one
decoder. Each keyframe interval (GOP) decodes independently, so the sampled
frames are grouped by GOP and the groups decoded on separate decoders in
threads (FFmpeg releases the GIL). Every frame is decoded from the same
keyframe as on one decoder, so the frames are bit-identical.
"""
from __future__ import annotations

import bisect
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from mstar.model.media_limits import VIDEO_DECODE_THREADS

# Concurrent decoders per request, each with VIDEO_DECODE_THREADS FFmpeg threads.
_MAX_DECODERS = max(1, min(6, (os.cpu_count() or 1) // VIDEO_DECODE_THREADS))
_pool: ThreadPoolExecutor | None = None


def _decode(filepath: str, indices: list[int]) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(filepath, num_ffmpeg_threads=VIDEO_DECODE_THREADS)
    return decoder.get_frames_at(indices=indices).data


def decode_frames_at(filepath: str, indices: list[int], decoder=None) -> torch.Tensor:
    """``[len(indices), C, H, W]`` uint8 frames at ``indices``, in that order.

    ``decoder``: an open torchcodec ``VideoDecoder`` on ``filepath`` to reuse
    for its keyframe index.
    """
    global _pool
    from torchcodec.decoders import VideoDecoder

    if decoder is None:
        decoder = VideoDecoder(filepath, num_ffmpeg_threads=VIDEO_DECODE_THREADS)
    # private in torchcodec 0.17; without it the clip decodes on one decoder
    key_frames = getattr(decoder, "_get_key_frame_indices", None)
    keys = key_frames().tolist() if key_frames is not None and _MAX_DECODERS > 1 else [0]
    order = sorted(set(indices))
    by_gop: dict[int, list[int]] = {}
    for i in order:
        by_gop.setdefault(bisect.bisect_right(keys, i), []).append(i)
    gops = list(by_gop.values())
    if len(gops) < 2:
        return decoder.get_frames_at(indices=indices).data
    workers = min(len(gops), _MAX_DECODERS)
    # contiguous runs of GOPs, balanced by frame count
    groups: list[list[int]] = [[] for _ in range(workers)]
    target, done = len(order) / workers, 0
    for gop in gops:
        groups[min(int((done + len(gop) / 2) / target), workers - 1)].extend(gop)
        done += len(gop)
    groups = [grp for grp in groups if grp]
    if _pool is None:
        _pool = ThreadPoolExecutor(_MAX_DECODERS, thread_name_prefix="video_decode")
    decoded = list(_pool.map(lambda grp: _decode(filepath, grp), groups))
    where = {i: (frames, r) for grp, frames in zip(groups, decoded, strict=True) for r, i in enumerate(grp)}
    out = torch.empty((len(indices), *decoded[0].shape[1:]), dtype=decoded[0].dtype)
    for k, i in enumerate(indices):
        frames, r = where[i]
        out[k] = frames[r]
    return out
