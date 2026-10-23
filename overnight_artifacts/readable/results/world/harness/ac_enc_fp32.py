"""AC encoder: fp32 weights vs bf16 weights (as served) vs bf16 autocast over fp32 weights (as upstream)."""
import sys

import torch
from torchcodec.decoders import VideoDecoder

sys.path.insert(0, sys.argv[1])
from mstar.model.registry import get_model_class, model_init_kwargs  # noqa: E402
from mstar.model.vjepa2.vjepa2_model import _preprocess_video  # noqa: E402

fr = VideoDecoder(sys.argv[2]).get_frames_in_range(0, 8).data.float().div(255)
x = _preprocess_video(fr, 256, 8).cuda()[None]
r = lambda a, b: ((a.float() - b.float()).norm() / a.float().norm()).item()
m32 = get_model_class("vjepa2_ac")(**model_init_kwargs("vjepa2_ac"))
s32 = m32.get_submodule("video_encoder", device="cuda", autocast_dtype=None)
mbf = get_model_class("vjepa2_ac")(**model_init_kwargs("vjepa2_ac"))
sbf = mbf.get_submodule("video_encoder", device="cuda", autocast_dtype=torch.bfloat16)
noise = 1e-3 * torch.randn_like(x)
with torch.inference_mode():
    f32 = s32._encode("prefill_video_rollout", x)
    f32n = s32._encode("prefill_video_rollout", x + noise)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ac = s32._encode("prefill_video_rollout", x)
        acn = s32._encode("prefill_video_rollout", x + noise)
    bf = sbf._encode("prefill_video_rollout", x)
    bfn = sbf._encode("prefill_video_rollout", x + noise)
print("fp32: noise 1e-3 ->", round(r(f32, f32n), 5))
print("fp32 weights + bf16 autocast (upstream style): vs fp32", round(r(f32, ac), 5), " noise ->", round(r(ac, acn), 5))
print("bf16 weights (served): vs fp32", round(r(f32, bf), 5), " noise ->", round(r(bf, bfn), 5))
