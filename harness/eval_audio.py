"""WER (Whisper large-v3, English normalizer) and speaker similarity (WavLM-base-plus-sv cosine to the
reference run's centroid) for collected runs. Imports nothing from mstar.
  eval_audio.py --ref DIR --runs name=DIR ... --out report.json"""
import argparse, json
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio, jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoFeatureExtractor, WavLMForXVector

ap = argparse.ArgumentParser()
ap.add_argument("--ref"); ap.add_argument("--runs", nargs="+"); ap.add_argument("--out")
a = ap.parse_args()
dev = "cuda"
proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).to(dev).eval()
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
norm = proc.tokenizer.normalize

def load16(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    return torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()

def run_dir(d):
    d = Path(d)
    reqs = {r["idx"]: r for r in map(json.loads, open(d / "requests.jsonl"))}
    rows = []
    for p in sorted((d / "wav").glob("*.wav")):
        idx = int(p.name.split("_")[0])
        rows.append((idx, p, reqs[idx]["text"]))
    hyps, embs, durs = {}, {}, {}
    for i in range(0, len(rows), 16):
        b = rows[i:i + 16]
        auds = [load16(p) for _, p, _ in b]
        feats = proc(auds, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
        with torch.no_grad():
            ids = asr.generate(feats, language="en", task="transcribe")
        for (idx, _, _), t in zip(b, proc.batch_decode(ids, skip_special_tokens=True)):
            hyps[idx] = t
        x = fe(auds, sampling_rate=16000, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            e = torch.nn.functional.normalize(sv(**x).embeddings, dim=-1).cpu().numpy()
        for (idx, _, _), ei, au in zip(b, e, auds):
            embs[idx], durs[idx] = ei, len(au) / 16000
    texts = {idx: t for idx, _, t in rows}
    return texts, hyps, embs, durs, len(reqs)

ref = run_dir(a.ref)
centroid = np.mean(list(ref[2].values()), 0); centroid /= np.linalg.norm(centroid)
report = {}
for spec in ["ref=" + a.ref] + a.runs:
    name, d = spec.split("=", 1)
    texts, hyps, embs, durs, n = ref if name == "ref" else run_dir(d)
    keys = sorted(hyps)
    refs_n = [norm(texts[k]) for k in keys]; hyp_n = [norm(hyps[k]) for k in keys]
    wer = jiwer.wer(refs_n, hyp_n)
    per = [jiwer.wer(r, h) for r, h in zip(refs_n, hyp_n)]
    if name == "ref":  # leave-one-out self similarity
        allv = np.stack([embs[k] for k in keys]); s = allv.sum(0)
        sims = [float(np.dot(allv[i], (s - allv[i]) / np.linalg.norm(s - allv[i]))) for i in range(len(keys))]
    else:
        sims = [float(np.dot(embs[k], centroid)) for k in keys]
    ratio = [durs[k] / ref[3][k] for k in keys if k in ref[3]]
    report[name] = {"n_requests": n, "n_audio": len(keys), "wer": wer, "wer_pct_items_gt_0.3": float(np.mean([p > 0.3 for p in per])),
                    "spk_sim_mean": float(np.mean(sims)), "spk_sim_p10": float(np.percentile(sims, 10)),
                    "dur_ratio_mean": float(np.mean(ratio)) if ratio else None,
                    "s_per_word": float(sum(durs.values()) / sum(len(texts[k].split()) for k in keys)),
                    "silent": int(sum(1 for k in keys if durs[k] < 0.3)),
                    "worst": sorted([(round(p, 2), k, hyps[k][:90]) for p, k in zip(per, keys)], reverse=True)[:5]}
    print(name, json.dumps({k: v for k, v in report[name].items() if k != "worst"}), flush=True)
Path(a.out).write_text(json.dumps(report, indent=1))
