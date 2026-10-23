"""Build the Qwen3-Omni workloads (CPU only; run by the runner's data phase).

  omni_a2t.jsonl   LibriSpeech test-clean (openslr/librispeech_asr, config "clean", split
                   "test"), first 200 utterances by id, written as 16 kHz mono WAV; prompt
                   "Transcribe the speech in this audio clip."; reference transcript kept
                   for WER; max_tokens 256, real EOS.
  omni_t2t.jsonl   the ShareGPT rows (same sampling as the Qwen3.5 text runs), max_tokens
                   = min(reply length, 256), real EOS (Qwen3-Omni's runner convention caps
                   the thinker at 256 tokens).
Text-to-speech uses benchmark.runner's own Qwen3-Omni T2S prompts (assets/simple_text_queries.txt).

  python omni_data.py --out /workspace/results/bench/_data/omni --sharegpt <q35 sharegpt.jsonl>
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    if not (out / "omni_a2t.jsonl").exists():
        import io

        import numpy as np
        import soundfile as sf
        from datasets import Audio, load_dataset
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
        ds = ds.cast_column("audio", Audio(decode=False))
        items = sorted(((r["id"], r) for r in ds), key=lambda x: x[0])[: a.n]
        rows = []
        for i, (uid, r) in enumerate(items):
            data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                raise SystemExit(f"{uid}: unexpected sample rate {sr}")
            path = out / "audio" / f"{i:03d}.wav"
            sf.write(path, data.astype(np.float32), 16000, subtype="PCM_16")
            rows.append({"id": f"ls{i:03d}", "utt": uid, "prompt": "Transcribe the speech in this audio clip.",
                         "audio": str(path), "reference": r["text"], "output_len": 256,
                         "seconds": round(len(data) / 16000, 3)})
        (out / "omni_a2t.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"omni_a2t: {len(rows)} utterances, {sum(r['seconds'] for r in rows):.0f} s of audio")
    if not (out / "omni_t2t.jsonl").exists():
        src = [json.loads(x) for x in Path(a.sharegpt).read_text().splitlines() if x.strip()]
        rows = [{**r, "output_len": min(int(r["output_len"]), 256)} for r in src]
        (out / "omni_t2t.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"omni_t2t: {len(rows)} rows")
    print("OMNI DATA DONE")


if __name__ == "__main__":
    main()
