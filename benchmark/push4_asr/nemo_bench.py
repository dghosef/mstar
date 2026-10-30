"""NeMo reference-package baseline for the NeMo-family ASR models: offline batched transcription of the same
200-utterance LibriSpeech set benchmark.asr_bench uses, at each batch size (the reference stack's own throughput
mode; compared with mstar's closed loop at concurrency = batch size). RTFx = audio seconds / wall seconds, median
of --repeats after one warm pass; hypotheses saved for WER with asr_bench's normalizer (asr_wer.py).

  CUDA_VISIBLE_DEVICES=2 ~/push4/engines/nemo/bin/python nemo_bench.py --model parakeet --out DIR [--batch 1 8 32 64]
"""
import argparse
import json
import statistics
import time
import wave
from pathlib import Path

import torch

HF = {"canary": "nvidia/canary-1b-v2", "canary_qwen": "nvidia/canary-qwen-2.5b",
      "parakeet": "nvidia/parakeet-tdt-0.6b-v3", "nemotron_asr": "nvidia/nemotron-3.5-asr-streaming-0.6b"}


def load(name):
    if name == "canary_qwen":
        from nemo.collections.speechlm2.models import SALM

        m = SALM.from_pretrained(HF[name])
    else:
        from nemo.collections.asr.models import ASRModel

        m = ASRModel.from_pretrained(HF[name])
    return m.to(torch.bfloat16 if name == "canary_qwen" else torch.float32).cuda().eval()


def _manifest(paths):
    path = Path(f"/tmp/nemo_manifest_{len(paths)}.json")
    with path.open("w") as f:
        for p in paths:
            with wave.open(p, "rb") as w:
                dur = w.getnframes() / w.getframerate()
            f.write(json.dumps({"audio_filepath": p, "duration": dur, "text": "", "lang": "en"}) + "\n")
    return str(path)


def transcribe(name, m, paths, bs):
    if name == "canary_qwen":
        prompts = [[{"role": "user", "content": f"Transcribe the following: {m.audio_locator_tag}", "audio": [p]}]
                   for p in paths]
        out = []
        for i in range(0, len(prompts), bs):
            ids = m.generate(prompts=prompts[i:i + bs], max_new_tokens=256)
            out += [m.tokenizer.ids_to_text(x.cpu()) for x in ids]
        return out
    kw = {"source_lang": "en", "target_lang": "en"} if name == "canary" else {}
    if name == "nemotron_asr":  # its prompt comes from each cut's language, which only a manifest carries
        paths = _manifest(paths)
    hyps = m.transcribe(paths, batch_size=bs, verbose=False, **kw)
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    return [h.text if hasattr(h, "text") else str(h) for h in hyps]


def main(a):
    d = Path(a.data) / "test_clean_200"
    refs = dict(line.split("\t", 1) for line in (d / "references.tsv").read_text().splitlines() if line.strip())
    paths = [str(d / f"{uid}.wav") for uid in refs]
    audio_s = 0.0
    for p in paths:
        with wave.open(p, "rb") as w:
            audio_s += w.getnframes() / w.getframerate()
    m = load(a.model)
    res = {"model": a.model, "hf": HF[a.model], "audio_s": audio_s, "dtype": "bf16" if a.model == "canary_qwen"
           else "fp32 (NeMo default; amp bf16 autocast)", "points": []}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        transcribe(a.model, m, paths[:16], 8)  # warm
        for bs in a.batch:
            walls, hyps = [], None
            for _ in range(a.repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                hyps = transcribe(a.model, m, paths, bs)
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - t0)
            wall = statistics.median(walls)
            res["points"].append({"batch": bs, "wall_s": round(wall, 3), "rtfx": round(audio_s / wall, 1),
                                  "hyps": dict(zip(refs, hyps, strict=True))})
            print(f"{a.model} batch {bs}: RTFx {audio_s / wall:.1f} (walls {[round(w, 2) for w in walls]})", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "nemo.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(HF))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path.home() / "push4/asr/data"))
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--repeats", type=int, default=2)
    main(ap.parse_args())
