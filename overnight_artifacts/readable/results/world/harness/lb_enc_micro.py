"""LingBot's per-chunk I2V condition encode: time per chunk and when the encoder stream turns steady."""
import sys
import time

import torch

arm = sys.argv[1]
sys.path.insert(0, arm)
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.model.lingbot_world.config import lingbot_world_1_3b  # noqa: E402
from mstar.model.lingbot_world.submodules import EncoderStream, encode_condition_chunk  # noqa: E402

cfg = lingbot_world_1_3b()
vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae",
                                       torch_dtype=torch.float32).eval().cuda()
torch.manual_seed(0)
img = torch.rand(3, 480, 832, device="cuda") * 2 - 1
import os
from mstar.model.components import wan_vae  # noqa: E402
bench = os.environ.get("ENC_BENCH") == "1"
cl = os.environ.get("ENC_CL") == "1"
if cl:
    vae = vae.to(memory_format=torch.channels_last_3d) if False else vae
outs = []
for rep in range(2):
    stream = EncoderStream(vae)
    for c in range(6):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with wan_vae.cudnn_benchmark(bench):
            y = encode_condition_chunk(vae, img, stream, c * 4, 4, cfg)
        torch.cuda.synchronize()
        if rep == 1:
            outs.append(y)
            print(f"chunk {c}: {1e3 * (time.perf_counter() - t):.1f} ms steady={stream.steady is not None}", flush=True)
torch.save(torch.stack([o.cpu() for o in outs]), sys.argv[2])
