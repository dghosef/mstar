"""NeMo Canary-1B-v2 on one WAV, CPU fp32: the decoder's top-5 tokens at the first generated step after the canary2
prompt (the same 9 tokens mstar sends), and the first 12 greedy tokens - shows whether the first word is a near-tie.

  nemo/bin/python canary_first_step.py <wav>
"""
import sys
import wave

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel

torch.set_num_threads(16)
m = ASRModel.from_pretrained("nvidia/canary-1b-v2", map_location="cpu").eval()
tok = m.tokenizer
with wave.open(sys.argv[1], "rb") as w:
    audio = torch.from_numpy(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768)
prompt = ["<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", "<|en|>", "<|en|>", "<|pnc|>",
          "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"]
ids = [tok.token_to_id(t) for t in prompt]
if len(sys.argv) > 2 and sys.argv[2] == "nemo-prompt":  # NeMo transcribe()'s prompt: a leading "▁" piece first
    ids = [16053] + ids
print("prompt ids:", ids)
with torch.inference_mode():
    feats, flen = m.preprocessor(input_signal=audio[None], length=torch.tensor([audio.numel()]))
    enc, elen = m.encoder(audio_signal=feats, length=flen)
    enc = enc.transpose(1, 2)
    enc_mask = torch.ones(enc.shape[:2], dtype=enc.dtype)
    if m.encoder_decoder_proj is not None if hasattr(m, "encoder_decoder_proj") else False:
        enc = m.encoder_decoder_proj(enc)
    seq = list(ids)
    for step in range(40):
        x = torch.tensor([seq])
        dec_mask = torch.ones_like(x, dtype=enc.dtype)
        hidden = m.transf_decoder(input_ids=x, decoder_mask=dec_mask, encoder_embeddings=enc, encoder_mask=enc_mask)
        logits = m.log_softmax(hidden_states=hidden)[0, -1] if hasattr(m, "log_softmax") else None
        top = torch.topk(logits.float(), 5)
        if step == 0:
            print("first step top-5:", [(tok.ids_to_tokens([int(i)])[0], round(float(v), 3)) for v, i in zip(*top, strict=True)])
        seq.append(int(top.indices[0]))
    print("greedy:", tok.ids_to_text(seq[len(ids):]))
