"""Masked padded tower == unpadded tower on short clips (GPU)."""
import sys

import torch

sys.path.insert(0, "/root/work/exp13")
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel  # noqa: E402

m = HiggsAudioModel("bosonai/higgs-audio-v3-stt")
enc = m.get_submodule("audio_encoder", device="cuda")
enc.audio_tower.float(); enc.projector.float()
dtype = torch.float32
g = torch.Generator(device="cuda").manual_seed(0)
worst = 0.0
with torch.inference_mode():
    widths = [37, 100, 151, 237, 300, 389, 399]
    feats = [torch.randn(1, 128, w, device="cuda", dtype=dtype, generator=g) for w in widths]
    padded = torch.cat([torch.nn.functional.pad(f, (0, 400 - f.shape[-1])) for f in feats])
    valid = torch.tensor(widths, device="cuda")
    batch = enc._encode(padded, valid)
    for i, (w, f) in enumerate(zip(widths, feats)):
        n = m.config.encoder_output_length(w)
        ref = enc._encode(f)[0, :n]
        got = batch[i, :n]
        rel = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        worst = max(worst, rel)
        print(w, n, ref.shape[0], f"rel max diff {rel:.4f}")
    full = torch.randn(3, 128, 400, device="cuda", dtype=dtype, generator=g)
    print("full width masked vs unmasked rel", ((enc._encode(full, torch.full((3,), 400, device="cuda")) - enc._encode(full)).abs().max() / enc._encode(full).abs().max()).item())
print("worst", worst)
