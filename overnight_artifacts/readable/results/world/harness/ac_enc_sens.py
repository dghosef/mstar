"""AC encoder sensitivity on a natural clip: eager B=1 vs B=4 rows, eager vs compile, tiny noise."""
import sys

import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder

sys.path.insert(0, sys.argv[1])
from mstar.model.registry import get_model_class, model_init_kwargs  # noqa: E402
from mstar.model.vjepa2.vjepa2_model import _preprocess_video  # noqa: E402

m = get_model_class("vjepa2_ac")(**model_init_kwargs("vjepa2_ac"))
sub = m.get_submodule("video_encoder", device="cuda", autocast_dtype=torch.bfloat16)
fr = VideoDecoder(sys.argv[2]).get_frames_in_range(0, 8).data.float().div(255)  # [8,3,H,W]
x = _preprocess_video(fr, 256, 8).cuda()[None]  # [1,8,3,256,256]
r = lambda a, b: ((a.float() - b.float()).norm() / a.float().norm()).item()
enc = lambda v: sub._encode("prefill_video_rollout", v)
with torch.inference_mode():
    e1 = enc(x)
    e4 = enc(x.expand(4, -1, -1, -1, -1).contiguous())
    print("eager B=4 rows vs B=1:", [round(r(e1[0], e4[i]), 5) for i in range(4)])
    c = torch.compile(lambda v: sub._encode("prefill_video_rollout", v))
    c1 = c(x)
    print("eager vs compiled B=1:", round(r(e1, c1), 5))
    c4 = torch.compile(lambda v: sub._encode("prefill_video_rollout", v))(x.expand(4, -1, -1, -1, -1).contiguous())
    print("compiled B=1 vs compiled B=4 row0:", round(r(c1[0], c4[0]), 5))
    for eps in (1e-4, 1e-3):
        print(f"input noise {eps}:", round(r(e1, enc(x + eps * torch.randn_like(x))), 5))
    raw = sub.encoder(x.reshape(8, 1, 3, 256, 256).repeat(1, 2, 1, 1, 1))
    print("encoder raw out: per-token std min %.3g median %.3g; max |x| %.3g" % (
        raw.float().std(-1).min().item(), raw.float().std(-1).median().item(), raw.float().abs().max().item()))
