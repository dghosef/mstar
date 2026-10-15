"""mstar's Qwen3.5 vision tower against transformers' on real video patches.

Both towers get the checkpoint's own vision weights and the same
``pixel_values_videos`` / ``video_grid_thw`` from the HF video processor (a
whole video, so ``t > 1``). mstar's tower runs its own patch embed, position
resample, rotary and merger; only its attention kernel is replaced by
per-frame SDPA, since the ragged FlashInfer resource needs the engine. So this
checks the port's numerics on video, not the served kernel.

Reported: max abs / mean abs error and cosine of the merged embeddings for
mstar vs transformers in fp32 and in bf16, with transformers bf16 vs fp32 as
the yardstick; plus mstar on the video as one ``(t, h, w)`` entry vs as ``t``
one-frame entries (the per-frame attention makes them the same computation).

  python vision_parity.py --video clip.mp4 [--model Qwen/Qwen3.8-27B]
"""
import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F


class PerFrameSDPA:
    """Stands in for the ragged attention resource: one segment per frame."""

    def __init__(self, seq_lengths):
        self.bounds = torch.tensor([0, *seq_lengths]).cumsum(0).tolist()

    def run(self, q, k, v):
        out = torch.empty_like(q)
        for a, b in zip(self.bounds[:-1], self.bounds[1:], strict=True):
            # [tokens, heads, dim] -> [heads, tokens, dim]
            qs, ks, vs = (t[a:b].transpose(0, 1) for t in (q, k, v))
            out[a:b] = F.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1)
        return out


def mstar_tower(weights_dir, dtype, device):
    from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
    from mstar.model.qwen3_5.config import Qwen3_5VisionConfig
    from mstar.model.qwen3_5.weight_loader import load_qwen3_5_vision_weights

    config = Qwen3_5VisionConfig.from_hf(weights_dir)
    torch.set_default_dtype(dtype)
    try:
        tower = Qwen3_5VisionModel(config).to(device)
    finally:
        torch.set_default_dtype(torch.float32)
    load_qwen3_5_vision_weights(tower, weights_dir, device=device)
    return tower.to(dtype).eval(), config


def run_mstar(tower, config, pixel_values, grid, device):
    from mstar.model.qwen3_5.components.vision import (
        vision_interpolation,
        vision_position_ids,
        vision_seq_lengths,
    )

    merge = config.spatial_merge_size
    indices, weights = vision_interpolation(grid, config.num_grid_per_side, merge, device)
    attn = PerFrameSDPA(vision_seq_lengths(grid))
    for block in tower.blocks:
        block.attn.ragged_attn = attn
    return tower(
        pixel_values.to(device), indices, weights, vision_position_ids(grid, merge, device),
    ).float()


def hf_tower(weights_dir, dtype, device):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    vision_config = AutoConfig.from_pretrained(weights_dir).vision_config
    vision_config._attn_implementation = "sdpa"
    tower = Qwen3_5VisionModel._from_config(vision_config, dtype=dtype).to(device)
    state = {}
    for path in glob.glob(os.path.join(weights_dir, "*.safetensors")):
        with safe_open(path, "pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual."):]] = f.get_tensor(key)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all("inv_freq" in m for m in missing), missing
    return tower.to(dtype).eval()


def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return {
        "max_abs": float((a - b).abs().max()),
        "mean_abs": float((a - b).abs().mean()),
        "cosine": float(F.cosine_similarity(a, b, dim=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="also write the results as JSON")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoVideoProcessor

    weights_dir = snapshot_download(args.model)
    processor = AutoVideoProcessor.from_pretrained(weights_dir)
    out = processor(videos=[args.video], return_tensors="pt")
    pixel_values, (grid_thw,) = out["pixel_values_videos"], out["video_grid_thw"]
    t, h, w = grid_thw.tolist()
    print(f"video grid (t, h, w) = {(t, h, w)}, {pixel_values.shape[0]} patches")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    results = {}
    with torch.inference_mode():
        hf = {}
        for dtype in (torch.float32, torch.bfloat16):
            tower = hf_tower(weights_dir, dtype, args.device)
            hf[dtype] = tower(
                pixel_values.to(args.device, dtype), grid_thw=grid_thw[None].to(args.device),
            ).pooler_output.float()
            del tower
        mstar = {}
        for dtype in (torch.float32, torch.bfloat16):
            tower, config = mstar_tower(weights_dir, dtype, args.device)
            mstar[dtype] = run_mstar(tower, config, pixel_values.to(dtype), [(t, h, w)], args.device)
            if dtype is torch.bfloat16:
                split = run_mstar(tower, config, pixel_values.to(dtype), [(1, h, w)] * t, args.device)
                results["mstar bf16: (t,h,w) vs t x (1,h,w)"] = compare(mstar[dtype], split)
            del tower
        results["mstar fp32 vs HF fp32"] = compare(mstar[torch.float32], hf[torch.float32])
        results["mstar bf16 vs HF bf16"] = compare(mstar[torch.bfloat16], hf[torch.bfloat16])
        results["mstar bf16 vs HF fp32"] = compare(mstar[torch.bfloat16], hf[torch.float32])
        results["yardstick: HF bf16 vs HF fp32"] = compare(hf[torch.bfloat16], hf[torch.float32])

    print("\n| comparison | max abs | mean abs | cosine |")
    print("|---|---:|---:|---:|")
    for name, r in results.items():
        print(f"| {name} | {r['max_abs']:.3e} | {r['mean_abs']:.3e} | {r['cosine']:.6f} |")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"grid": [t, h, w], "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
