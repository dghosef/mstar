#!/usr/bin/env python3
"""Offline competitor for LingBot-World-V2 1.3B causal-fast: the official repo (Robbyant/lingbot-world-v2,
wan.WanI2VCausal, causal_fast, local attention 18 / sink 6, chunk 4 latents, 4 distilled steps at the
config's shift), run as M*'s parity oracle runs it (test/lingbot_world/record_oracle.py: the M* venv plus the
repo on sys.path, float32 matmul precision 'high'). 480x832, 93 frames, seed 42, offload_model=False, the
same example (examples/03 image, poses, intrinsics) and prompt as the M* arm. It decodes the whole clip at
the end (no first-chunk latency): only total time / fps is comparable. Batch is 1 (single request by design,
like M*'s max_concurrent_requests: 1). Saves the last 16 frames (uint8 RGB) for the gate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doff_common as D  # noqa: E402


def extra(ap):
    ap.add_argument("--repo", default="/root/work/engines/lingbot-world-v2")
    ap.add_argument("--ckpt", default="/root/work/hf/lb13b")
    ap.add_argument("--assets", default="/root/work/hf/lb14b_assets")


def main():
    a = D.parse(extra)
    for p in filter(None, os.environ.get("DGEN_OVERLAY", "").split(":")):
        sys.path.insert(0, p)
    sys.path.insert(0, a.repo)
    import torch
    torch.set_float32_matmul_precision("high")
    import wan
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    cfg = WAN_CONFIGS["i2v-1.3B"]
    pipe = wan.WanI2VCausal(config=cfg, checkpoint_dir=a.ckpt, device_id=0, rank=0, local_attn_size=18, sink_size=6,
                            infer_mode="causal_fast", assets_dir=a.assets)

    def process(batch):
        outs = []
        for _, it, _ in batch:
            img = Image.open(it["image_path"]).convert("RGB")
            v = pipe.generate(it["prompt"], img, action_path=it["example_dir"], chunk_size=4,
                              max_area=MAX_AREA_CONFIGS["480*832"], frame_num=93, shift=cfg.sample_shift, seed=42,
                              offload_model=False)  # [3, F, H, W] in [-1, 1]
            fr = ((v.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).contiguous().cpu()
            outs.append(fr[-16:].numpy().tobytes())
        return outs

    D.run(a, process, sync=torch.cuda.synchronize)


if __name__ == "__main__":
    main()
