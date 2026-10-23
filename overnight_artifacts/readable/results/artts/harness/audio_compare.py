#!/usr/bin/env python3
"""Compare two directories of <id>.wav files from tts_speech_bench.

    python audio_compare.py A_dir B_dir

Per file: byte-identical PCM, equal length, and SNR of B against A where the
lengths match. Prints a summary line.
"""
import sys
from pathlib import Path

import numpy as np


def pcm(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes()[44:], dtype=np.int16).astype(np.float64)


a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(p.name for p in a_dir.glob("*.wav") if (b_dir / p.name).exists())
ident = same_len = 0
snrs, len_ratio = [], []
for n in names:
    a, b = pcm(a_dir / n), pcm(b_dir / n)
    len_ratio.append(len(b) / max(len(a), 1))
    if len(a) == len(b):
        same_len += 1
        if np.array_equal(a, b):
            ident += 1
        else:
            err = np.sum((a - b) ** 2)
            snrs.append(10 * np.log10(np.sum(a ** 2) / max(err, 1e-9)))
print(f"{len(names)} files: identical={ident} same_len={same_len} "
      f"snr_median={np.median(snrs) if snrs else float('nan'):.1f}dB "
      f"len_ratio_mean={np.mean(len_ratio):.3f} total_audio_ratio="
      f"{sum(len(pcm(b_dir / n)) for n in names) / max(1, sum(len(pcm(a_dir / n)) for n in names)):.3f}")
