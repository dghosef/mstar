#!/usr/bin/env python3
"""diffusers in-process arm: the reference for the correctness gate, and the timed
"diffusers (tuned)" arm. Never imports mstar.

  gate  B=1 over a prompt slice with seeds seed_base + i (CPU generator, like mstar and its
        oracles), saving each output: <out>/<i>.png, or for video <i>.npy (uint8 frames
        T x H x W x 3, never through a codec) plus <i>.mp4 for the gallery.
  time  B=1 latency and offline-batch throughput: for each batch size b, `num` images
        (videos) in batches of b, after one untimed warmup batch of that size. Timing is
        the pipeline call (text encode -> denoise -> VAE decode -> PIL / uint8), with
        torch.cuda.synchronize, no HTTP, no PNG/mp4 encode.

Knobs: --attn (diffusers attention backend name, or `default` = torch SDPA dispatch),
--compile none|regional|full (regional = transformer.compile_repeated_blocks(fullgraph=True),
the diffusers docs' recommended path; full = torch.compile(transformer, fullgraph=True,
mode=max-autotune-no-cudagraphs)), --compile-vae (torch.compile the VAE decode).
float32 matmul precision "high" (TF32), as mstar's engine sets process-wide and as its
oracle recorders do.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

MODELS = {
    "zimage": dict(hf="Tongyi-MAI/Z-Image-Turbo", kind="image"),
    "klein4b": dict(hf="black-forest-labs/FLUX.2-klein-4B", kind="image"),
    "klein9b": dict(hf="black-forest-labs/FLUX.2-klein-9B", kind="image"),
    "wan22": dict(hf="Wan-AI/Wan2.2-TI2V-5B-Diffusers", kind="video"),
}


def load(model: str, attn: str, compile_mode: str, compile_vae: bool, dynamic=None):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(MODELS[model]["hf"], torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if attn != "default":
        pipe.transformer.set_attention_backend(attn)
    if compile_mode != "none":
        # prompts of different lengths change the token count; a generous recompile budget keeps dynamo from
        # giving up (dynamic=True compiles one symbolic-shape graph instead of one per length)
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 4096
    if compile_mode == "regional":
        pipe.transformer.compile_repeated_blocks(fullgraph=True, dynamic=dynamic)
    elif compile_mode == "full":
        pipe.transformer = torch.compile(pipe.transformer, fullgraph=True, mode="max-autotune-no-cudagraphs",
                                         dynamic=dynamic)
    if compile_vae:
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune-no-cudagraphs")
    return pipe


def call(pipe, kind, prompts, seeds, a):
    gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
    kw = dict(prompt=prompts, height=a.height, width=a.width, num_inference_steps=a.steps,
              guidance_scale=a.guidance, generator=gens if len(gens) > 1 else gens[0])
    if kind == "video":
        kw.update(num_frames=a.frames, negative_prompt=[a.negative_prompt] * len(prompts), output_type="np")
        frames = pipe(**kw).frames  # B x T x H x W x 3 float in [0, 1]
        return [np.asarray(np.clip(f, 0, 1) * 255).round().astype(np.uint8) for f in frames]
    return pipe(**kw).images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "time"])
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--guidance", type=float, required=True)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--attn", default="default")
    ap.add_argument("--compile", default="none", choices=["none", "regional", "full"])
    ap.add_argument("--compile-vae", action="store_true")
    ap.add_argument("--dynamic", action="store_true", help="compile with dynamic=True (symbolic shapes)")
    ap.add_argument("--count", type=int, default=64, help="gate: prompts 0..count-1")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--batch-sizes", default="1", help="time: comma list")
    ap.add_argument("--num", default="16", help="time: images per batch size (comma list or one value)")
    ap.add_argument("--prompt-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kind = MODELS[a.model]["kind"]
    prompts = [p for p in Path(a.prompts).read_text().splitlines() if p.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    pipe = load(a.model, a.attn, a.compile, a.compile_vae, True if a.dynamic else None)
    meta = {"args": vars(a), "load_s": round(time.time() - t_load, 1), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "matmul_precision": torch.get_float32_matmul_precision()}
    import diffusers
    import transformers
    meta.update(diffusers=diffusers.__version__, transformers=transformers.__version__)

    if a.mode == "gate":
        lat = []
        for i in range(a.count):
            p, s = prompts[i % len(prompts)], a.seed_base + i
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = call(pipe, kind, [p], [s], a)[0]
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            if kind == "image":
                res.save(out / f"{i:03d}.png")
            else:
                np.save(out / f"{i:03d}.npy", res)
                import imageio.v3 as iio
                iio.imwrite(out / f"{i:03d}.mp4", res, fps=a.fps, codec="libx264")
            print(f"gate {i} seed {s} {lat[-1]:.2f}s", flush=True)
        meta["latency_s"] = lat
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        return

    sizes = [int(x) for x in a.batch_sizes.split(",")]
    nums = [int(x) for x in a.num.split(",")]
    nums = nums if len(nums) == len(sizes) else nums * len(sizes)
    rows = []
    k = a.prompt_offset
    for b, n in zip(sizes, nums):
        try:
            wp = [prompts[(100000 + j) % len(prompts)] for j in range(b)]
            call(pipe, kind, wp, [7 + j for j in range(b)], a)  # warmup batch (compile / autotune for b)
            lat, done = [], 0
            torch.cuda.synchronize()
            t_all = time.perf_counter()
            while done < n:
                bb = min(b, n - done)
                ps = [prompts[(k + j) % len(prompts)] for j in range(bb)]
                ss = [a.seed_base + done + j for j in range(bb)]
                t0 = time.perf_counter()
                res = call(pipe, kind, ps, ss, a)
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)
                assert len(res) == bb
                done += bb
                k += bb
            wall = time.perf_counter() - t_all
            s = sorted(lat)
            rows.append({"batch": b, "num": n, "completed": n, "failed": 0, "wall_s": wall,
                         "throughput_per_s": n / wall, "throughput_per_min": 60 * n / wall,
                         "batch_lat_mean": statistics.fmean(lat), "batch_lat_p50": statistics.median(lat),
                         "batch_lat_min": s[0], "batch_lat_max": s[-1], "batch_latencies": lat,
                         "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30})
        except torch.OutOfMemoryError as e:
            rows.append({"batch": b, "num": n, "completed": 0, "failed": n, "error": f"OOM: {str(e)[:200]}"})
            torch.cuda.empty_cache()
        print(json.dumps({k2: v for k2, v in rows[-1].items() if k2 != "batch_latencies"}), flush=True)
    meta["rows"] = rows
    (out / "results.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
