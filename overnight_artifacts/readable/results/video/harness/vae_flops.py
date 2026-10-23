"""FLOPs and achieved TF/s of the fast Wan2.2 VAE decode at a size."""
import sys
import time

import torch
from diffusers import AutoencoderKLWan
from torch.utils.flop_counter import FlopCounterMode

from mstar.model.components import wan_vae

t, h, w = (int(x) for x in sys.argv[1:4])
vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae").eval().to("cuda", torch.bfloat16)
z = torch.randn(1, 48, t, h, w, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    with torch.inference_mode(False):
        wan_vae.to_channels_last(vae)
    step = wan_vae.DecoderStep(vae, compile=False)
    with FlopCounterMode(display=False) as fc:
        wan_vae.decode_causal(vae, z, wan_vae.new_feat_cache(vae), step, clip_start=True)
    flops = fc.get_total_flops()
    step = wan_vae.DecoderStep(vae, compile=True, deterministic=True)
    for _ in range(2):
        wan_vae.decode_causal(vae, z, wan_vae.new_feat_cache(vae), step, clip_start=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wan_vae.decode_causal(vae, z, wan_vae.new_feat_cache(vae), step, clip_start=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
print(f"latent {t}x{h}x{w}: {flops / 1e12:.1f} TFLOP, {dt * 1000:.0f} ms, {flops / dt / 1e12:.0f} TF/s")
