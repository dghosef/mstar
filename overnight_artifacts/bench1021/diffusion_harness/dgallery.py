#!/usr/bin/env python3
"""Human-check contact sheet: rows = prompts, columns = arms (same prompt and seed), tiles downscaled, one JPEG.

  python dgallery.py --root <out>/gate --kind image --tags ref_a,mstar_pr,... --items 0,8,16,24 --out sheet.jpg
Video tiles show the middle frame (decoded from each arm's mp4; the reference's raw frames).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def tile(root: Path, tag: str, i: int, kind: str, size: int):
    d = root / tag
    if kind == "image":
        p = d / f"{i:03d}.png"
        if not p.exists():
            return None
        im = Image.open(p).convert("RGB")
    else:
        if (d / f"{i:03d}.npy").exists():
            v = np.load(d / f"{i:03d}.npy", mmap_mode="r")
            im = Image.fromarray(np.asarray(v[len(v) // 2]))
        elif (d / f"{i:03d}.mp4").exists():
            import av
            with av.open(str(d / f"{i:03d}.mp4")) as c:
                frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            im = Image.fromarray(frames[len(frames) // 2])
        else:
            return None
    im.thumbnail((size, size))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--items", default="0,8,16,24")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, tags, items = Path(a.root), a.tags.split(","), [int(x) for x in a.items.split(",")]
    tiles = {(t, i): tile(root, t, i, a.kind, a.size) for t in tags for i in items}
    w = max((im.width for im in tiles.values() if im), default=a.size)
    h = max((im.height for im in tiles.values() if im), default=a.size)
    head = 16
    sheet = Image.new("RGB", (w * len(tags), head + h * len(items)), "white")
    dr = ImageDraw.Draw(sheet)
    for c, t in enumerate(tags):
        dr.text((c * w + 3, 2), t[:28], fill="black")
        for r, i in enumerate(items):
            im = tiles[(t, i)]
            if im:
                sheet.paste(im, (c * w, head + r * h))
    q = 80
    while True:
        sheet.save(a.out, quality=q)
        if Path(a.out).stat().st_size <= 200_000 or q <= 30:
            break
        q -= 10
    print(a.out, Path(a.out).stat().st_size)


if __name__ == "__main__":
    main()
