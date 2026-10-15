"""What mstar's video intake costs per request, and what preprocessing on the GPU would.

For the first N items of each workload (``prepare_videos.py`` layout), on one
GPU with no server running:

- ``decode``: `Qwen3_5DenseModel.load_video` (torchcodec, sampled frames only);
- ``preprocess cpu``: `_preprocess_video` as the API server runs it, with the
  model's ``PREPROCESS_TORCH_THREADS``;
- ``preprocess gpu``: the same video processor call with ``device="cuda"``,
  its max abs difference from the CPU patches, and whether the grid matches.

  python video_cpu_cost.py --data ~/video_bench/data --items 8
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers.video_utils import VideoMetadata

    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = Qwen3_5DenseModel(snapshot_download(args.model))
    torch.set_num_threads(Qwen3_5DenseModel.PREPROCESS_TORCH_THREADS)
    processor = model.video_processor

    print("| workload | frames | grid (t,h,w) | merged tokens | decode ms | preprocess cpu ms "
          "| preprocess gpu ms | gpu vs cpu max abs | grid equal |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---|")
    for wl in ("short", "long"):
        root = Path(args.data) / wl
        rows = [json.loads(line) for line in open(root / "data/items.jsonl")][: args.items]
        stats = {"decode": [], "cpu": [], "gpu": []}
        for row in rows:
            path = str(root / row["video_path"])
            t0 = time.perf_counter()
            loaded = model.load_video(path, "cpu")
            t1 = time.perf_counter()
            video = model._preprocess_video(loaded.data, loaded.metadata)
            t2 = time.perf_counter()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            gpu = processor(
                videos=[loaded.data.cuda()], video_metadata=[VideoMetadata(**loaded.metadata)],
                do_sample_frames=False, return_tensors="pt", device="cuda",
            )
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            diff = float((gpu["pixel_values_videos"].cpu() - video.pixel_values).abs().max())
            same_grid = torch.equal(gpu["video_grid_thw"][0].cpu(), video.grid)
            stats["decode"].append((t1 - t0) * 1e3)
            stats["cpu"].append((t2 - t1) * 1e3)
            stats["gpu"].append((t4 - t3) * 1e3)
            t, h, w = video.grid.tolist()
            print(f"| {wl} | {loaded.data.shape[0]} | {(t, h, w)} | {t * h * w // 4} | {stats['decode'][-1]:.0f} "
                  f"| {stats['cpu'][-1]:.0f} | {stats['gpu'][-1]:.0f} | {diff:.2e} | {same_grid} |")
        print(f"| **{wl} median** | | | | {statistics.median(stats['decode']):.0f} "
              f"| {statistics.median(stats['cpu']):.0f} | {statistics.median(stats['gpu']):.0f} | | |")


if __name__ == "__main__":
    main()
