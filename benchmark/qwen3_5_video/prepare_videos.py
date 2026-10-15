"""Build the two video workloads as local Video-MME-layout directories.

``long``: whole Video-MME "short"-duration videos (1-3 min, real footage),
one multiple-choice question each. ``short``: ~6 s clips cut from the same
videos at different offsets and re-encoded, so every clip is distinct content
and no engine's multimodal cache can serve a request from an earlier one.

Both are read by ``benchmark.runner --dataset video_mme --video-mme-dir DIR``
(``DIR/data/*.jsonl`` + ``DIR/videos/*.mp4``). The mirror's own jsonl is not
UTF-8 and stores ``options`` as a string, so rows are rewritten here.

  python prepare_videos.py --out ~/video_bench/data --videos 16 --clips-per-video 15
"""
import argparse
import ast
import json
import subprocess
from pathlib import Path

REPO = "zhaochenyang20/Video_MME"
CLIP_SECONDS = 6


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", type=int, default=16)
    ap.add_argument("--clips-per-video", type=int, default=15)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    out = Path(args.out).expanduser()
    rows, seen = [], set()
    for part in ("0000", "0200", "0400"):
        path = hf_hub_download(REPO, f"data/test_part_{part}.jsonl", repo_type="dataset")
        for line in open(path, encoding="latin-1"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("duration") != "short" or row["videoID"] in seen:
                continue
            seen.add(row["videoID"])
            rows.append(row)
    rows = rows[: args.videos]

    long_dir, short_dir = out / "long", out / "short"
    for d in (long_dir, short_dir):
        (d / "videos").mkdir(parents=True, exist_ok=True)
        (d / "data").mkdir(parents=True, exist_ok=True)

    long_rows, short_rows = [], []
    for row in rows:
        src = Path(hf_hub_download(REPO, row["video_path"], repo_type="dataset"))
        name = f"{row['videoID']}.mp4"
        dst = long_dir / "videos" / name
        if not dst.exists():
            dst.symlink_to(src)
        options = row["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        long_rows.append({"question": row["question"], "options": list(options),
                          "video_path": f"videos/{name}"})

        total = duration(src)
        step = max((total - CLIP_SECONDS) / args.clips_per_video, 1.0)
        for k in range(args.clips_per_video):
            start = k * step
            if start + CLIP_SECONDS > total:
                break
            clip = short_dir / "videos" / f"{row['videoID']}_{k:02d}.mp4"
            if not clip.exists():
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", str(CLIP_SECONDS),
                     "-i", str(src), "-an", "-c:v", "libx264", "-preset", "veryfast",
                     "-pix_fmt", "yuv420p", str(clip)],
                    check=True,
                )
            short_rows.append({"question": "Describe what happens in this clip.", "options": [],
                               "video_path": f"videos/{clip.name}"})

    # interleave sources so consecutive requests are different videos
    short_rows.sort(key=lambda r: (r["video_path"].rsplit("_", 1)[1], r["video_path"]))
    for d, items in ((long_dir, long_rows), (short_dir, short_rows)):
        with open(d / "data" / "items.jsonl", "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"{d}: {len(items)} items")


if __name__ == "__main__":
    main()
