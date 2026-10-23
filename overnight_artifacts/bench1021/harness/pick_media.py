"""Pick a small media sample from a bench node's results, so a reader can see and hear each engine's outputs
after the volumes are deleted: from each model's gate/ and gallery/ folders, the first file of each arm's output
folder (image, video or audio), at most PER_MODEL files and PER_MODEL_MB per model, no file over MAX_FILE_MB.
Prints the chosen paths relative to the results root (for rsync --files-from) and a size summary on stderr.

  python3 pick_media.py /workspace/results/bench > picked.txt
"""
import sys
from pathlib import Path

MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".wav", ".mp3", ".flac"}
PER_MODEL, PER_MODEL_MB, MAX_FILE_MB = 12, 40, 20


def main():
    root = Path(sys.argv[1])
    total = 0
    for model in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        by_dir = {}
        for sub in ("gate", "gate_t2s", "gallery"):
            for p in sorted((model / sub).rglob("*")) if (model / sub).is_dir() else []:
                if p.suffix.lower() in MEDIA and p.stat().st_size <= MAX_FILE_MB << 20:
                    by_dir.setdefault(p.parent, p)  # first file of each output folder (one per arm/knob)
        picked, mb = [], 0.0
        for p in by_dir.values():
            size = p.stat().st_size / 2**20
            if len(picked) >= PER_MODEL or mb + size > PER_MODEL_MB:
                break
            picked.append(p)
            mb += size
        for p in picked:
            print(p.relative_to(root))
        if picked:
            print(f"{model.name}: {len(picked)} files, {mb:.1f} MB", file=sys.stderr)
        total += mb
    print(f"total {total:.0f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
