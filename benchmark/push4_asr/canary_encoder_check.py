"""Canary window check on CPU, fp32: log-mel features and encoder output of one WAV from mstar (unpadded, and padded to
the encoder's capture bucket as the served path pads) and from NeMo; run once per side, then compare.

  PYTHONPATH=<tree> venv/bin/python canary_encoder_check.py mstar <wav> <out.pt>
  nemo/bin/python canary_encoder_check.py nemo <wav> <out.pt>
  python canary_encoder_check.py cmp <mstar.pt> <nemo.pt>
"""
import sys
import wave

import numpy as np
import torch

torch.set_num_threads(16)
side = sys.argv[1]


def load(path):
    with wave.open(path, "rb") as w:
        return torch.from_numpy(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768)


if side == "mstar":
    from mstar.model.canary.canary_model import CanaryModel
    from mstar.model.components.fastconformer_submodule import _bucket_for

    sub = CanaryModel("nvidia/canary-1b-v2").get_submodule("audio_encoder", device="cpu", autocast_dtype=None)
    audio = load(sys.argv[2])
    with torch.inference_mode():
        feats = sub.log_mel.batch([audio])[0]
        n = feats.shape[0]
        out_len = int(sub.encoder.config.output_lengths(n))
        plain, _ = sub.encoder(feats[None], torch.tensor([n]))
        bucket = _bucket_for(n)
        padded_feats = torch.nn.functional.pad(feats, (0, 0, 0, bucket - n))
        padded, _ = sub.encoder(padded_feats[None], torch.tensor([n]))
    torch.save({"feats": feats, "plain": plain[0, :out_len], "padded": padded[0, :out_len], "bucket": bucket},
               sys.argv[3])
    print(f"mstar: {n} mel frames -> bucket {bucket}, {out_len} encoder frames")
elif side == "nemo":
    from nemo.collections.asr.models import ASRModel

    m = ASRModel.from_pretrained("nvidia/canary-1b-v2", map_location="cpu").eval()
    audio = load(sys.argv[2])
    with torch.inference_mode():
        feats, flen = m.preprocessor(input_signal=audio[None], length=torch.tensor([audio.numel()]))
        enc, elen = m.encoder(audio_signal=feats, length=flen)
    torch.save({"feats": feats[0].T[:int(flen)], "enc": enc[0].T[:int(elen)]}, sys.argv[3])
    print(f"nemo: {int(flen)} mel frames, {int(elen)} encoder frames")
else:
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])

    def rel(x, y):
        n = min(len(x), len(y))
        x, y = x[:n].float(), y[:n].float()
        d = (x - y).norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-6)
        return (f"frames {len(x)} vs {len(y)}: rel err mean {d.mean():.2e} max {d.max():.2e} "
                f"(frame {int(d.argmax())} of {n})")

    print("features  mstar vs nemo :", rel(a["feats"], b["feats"]))
    print("encoder   mstar unpadded vs nemo :", rel(a["plain"], b["enc"]))
    print("encoder   mstar padded   vs nemo :", rel(a["padded"], b["enc"]))
    print("encoder   mstar padded vs unpadded :", rel(a["padded"], a["plain"]))
