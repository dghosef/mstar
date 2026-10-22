"""PSNR / exactness between saved uint8 frame arrays: psnr.py ref.npy other.npy [...]"""
import math
import sys

import numpy as np

ref = np.load(sys.argv[1])
for p in sys.argv[2:]:
    o = np.load(p)
    if o.shape != ref.shape:
        print(p, "shape", o.shape, "vs", ref.shape)
        continue
    d = o.astype(np.float64) - ref.astype(np.float64)
    mse = float((d ** 2).mean())
    per = [(10 * math.log10(255 ** 2 / m) if m > 0 else math.inf) for m in (d ** 2).reshape(len(d), -1).mean(1)]
    print(p, "bit_exact" if mse == 0 else f"psnr={10 * math.log10(255 ** 2 / mse):.2f}dB min_frame={min(per):.2f}dB "
          f"max_abs={int(np.abs(d).max())} frac_diff={float((d != 0).mean()):.5f}")
