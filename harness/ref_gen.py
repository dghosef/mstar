"""Reference outputs from the official qwen-tts package (refenv). Imports nothing from mstar.
bf16 + sdpa, the package's own sampling defaults (temp 0.9, top-k 50, top-p 1.0, rep 1.05), max 2048."""
import argparse, json, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

ap = argparse.ArgumentParser()
ap.add_argument("--texts"); ap.add_argument("--out"); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--speaker", default="Ryan"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out); (out / "wav").mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed)
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16,
                                  attn_implementation="sdpa")
texts = [json.loads(l) for l in open(a.texts)]
recs = []
for i in range(0, len(texts), a.batch):
    chunk = texts[i:i + a.batch]
    t0 = time.time()
    wavs, sr = m.generate_custom_voice(text=[t["text"] for t in chunk], speaker=[a.speaker] * len(chunk),
                                       language=["English"] * len(chunk), max_new_tokens=2048)
    for j, (t, w) in enumerate(zip(chunk, wavs)):
        idx = i + j
        sf.write(out / "wav" / f"{idx:04d}_{t['id']}.wav", np.asarray(w, dtype=np.float32), sr)
        recs.append({"idx": idx, "tid": t["id"], "text": t["text"], "audio_s": len(w) / sr, "ok": True})
    print(f"{i + len(chunk)}/{len(texts)} {time.time() - t0:.1f}s", flush=True)
with open(out / "requests.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
