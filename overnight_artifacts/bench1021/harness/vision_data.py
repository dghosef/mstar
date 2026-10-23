"""Build the Qwen3.5 vision workloads once per node (CPU only; run by the runner's data phase).

Preprocessing limits, identical for every engine (fixed here, client side, so no engine's
own resize or frame sampler changes the input):
  images  MMMU validation (lmms-lab/MMMU), single-image multiple-choice questions, sorted
          by id, first 200. Each image is scaled to fit 896x896 (aspect kept), then each
          side is rounded down to a multiple of 224 (min 224): 224 = lcm of the 28 px (SGLang)
          and 32 px (HF / vLLM / mstar) resize factors, so no engine resizes it again.
  videos  Video-MME "short" videos (zhaochenyang20/Video_MME mirror). Clips of 8, 30 and
          60 s, re-encoded at exactly 2 fps (every engine's Qwen3-VL default), 448x224
          (scale to cover, centre crop), no audio: 16 / 60 / 120 frames, which every engine
          samples in full (all under the per-frame and total pixel limits of HF, vLLM,
          SGLang and mstar). Several clips per source video at different offsets, so every
          request in a cell is a distinct file.

Outputs under --out:
  mmmu.jsonl         rows {id, prompt, image, answer, output_len=256}   (throughput + MCQ eval)
  video_8s.jsonl, video_30s.jsonl, video_60s.jsonl  rows {id, prompt, video, output_len=256}
  prompts_vision_img.jsonl / prompts_vision_video.jsonl   parity prompts (parity.py format)

  python vision_data.py --out /workspace/results/bench/_data/vision
"""
import argparse
import ast
import json
import subprocess
from pathlib import Path

VIDEO_REPO = "zhaochenyang20/Video_MME"
LETTERS = "ABCDEFGHIJ"
INSTRUCT = "Answer with the option's letter from the given choices directly."


def fit_224(w, h, box=896):
    s = min(box / w, box / h, 1.0)
    w2, h2 = max(224, int(w * s) // 224 * 224), max(224, int(h * s) // 224 * 224)
    return w2, h2


def mmmu(out: Path, n: int):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/MMMU", split="validation")
    keep = []
    for r in ds:
        if r.get("question_type") != "multiple-choice" or r.get("image_2") is not None or r.get("image_1") is None:
            continue
        try:
            opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else list(r["options"])
        except (ValueError, SyntaxError):
            continue
        if not 2 <= len(opts) <= len(LETTERS) or r["question"].count("<image 1>") > 1:
            continue
        keep.append((r["id"], r, opts))
    keep.sort(key=lambda x: x[0])
    keep = keep[:n]
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (rid, r, opts) in enumerate(keep):
        img = r["image_1"].convert("RGB")
        img = img.resize(fit_224(*img.size))
        path = out / "img" / f"{i:03d}.png"
        img.save(path)
        q = r["question"].replace("<image 1>", "the image").strip()
        prompt = q + "\n" + "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts)) + "\n" + INSTRUCT
        rows.append({"id": f"mmmu{i:03d}", "mmmu_id": rid, "prompt": prompt, "image": str(path),
                     "answer": r["answer"], "size": list(img.size), "output_len": 256})
    (out / "mmmu.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"mmmu: {len(rows)} single-image MCQ rows")
    return rows


def duration(path: Path) -> float:
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       check=True, capture_output=True, text=True)
    return float(o.stdout.strip())


def videos(out: Path, n_src: int, per_src: int, lengths=(8, 30, 60)):
    from huggingface_hub import hf_hub_download
    srcs, seen = [], set()
    for part in ("0000", "0200", "0400"):
        p = hf_hub_download(VIDEO_REPO, f"data/test_part_{part}.jsonl", repo_type="dataset")
        for line in open(p, encoding="latin-1"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("duration") != "short" or row["videoID"] in seen:
                continue
            seen.add(row["videoID"])
            srcs.append(row)
    srcs = srcs[:n_src]
    (out / "vid").mkdir(parents=True, exist_ok=True)
    rows = {L: [] for L in lengths}
    for row in srcs:
        src = Path(hf_hub_download(VIDEO_REPO, row["video_path"], repo_type="dataset"))
        total = duration(src)
        for L in lengths:
            if total < L + 1:
                continue
            step = (total - L) / per_src
            for k in range(per_src):
                start = k * step
                clip = out / "vid" / f"{row['videoID']}_{L}s_{k}.mp4"
                if not clip.exists():
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", str(L), "-i", str(src),
                                    "-an", "-vf", "fps=2,scale=448:224:force_original_aspect_ratio=increase,crop=448:224",
                                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                    str(clip)], check=True)
                rows[L].append({"id": f"v{L}_{row['videoID']}_{k}", "prompt": "Describe what happens in this video in detail.",
                                "video": str(clip), "output_len": 256})
    for L, rs in rows.items():
        # interleave sources so consecutive requests come from different videos
        rs.sort(key=lambda r: (r["id"].rsplit("_", 1)[1], r["id"]))
        (out / f"video_{L}s.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rs))
        print(f"video_{L}s: {len(rs)} clips")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmmu", type=int, default=200)
    ap.add_argument("--video-sources", type=int, default=24)
    ap.add_argument("--clips-per-source", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = mmmu(out, a.mmmu) if not (out / "mmmu.jsonl").exists() else \
        [json.loads(x) for x in (out / "mmmu.jsonl").read_text().splitlines()]
    if not all((out / f"video_{L}s.jsonl").exists() for L in (8, 30, 60)):
        videos(out, a.video_sources, a.clips_per_source)
    vids = {L: [json.loads(x) for x in (out / f"video_{L}s.jsonl").read_text().splitlines()] for L in (8, 30, 60)}
    img_prompts = [{"text": r["prompt"], "image": r["image"]} for r in rows[:8]]
    vid_prompts = [{"text": r["prompt"], "video": r["video"]} for L in (8, 30, 60) for r in vids[L][:2]]
    (out / "prompts_vision_img.jsonl").write_text("".join(json.dumps(x) + "\n" for x in img_prompts))
    (out / "prompts_vision_video.jsonl").write_text("".join(json.dumps(x) + "\n" for x in vid_prompts))
    print("VISION DATA DONE")


if __name__ == "__main__":
    main()
