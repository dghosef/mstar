#!/usr/bin/env python3
"""Sanity gate for the dgen entries (run in the diffusers venv, GPU 7). Criteria are fixed here, before
any output is collected. Each tag's outputs: <root>/<tag>/<workload>/outputs.jsonl (texts, request i ->
item index) or <root>/<tag>/<workload>/out/<i>.<ext> (binary), plus DONE ({"failed": n}).
Tags: mstar_pr, mstar_pr_b (a second M* PR process: the same-code control), mstar_main, <engine>__<knob>.

Sanity gate (user direction 2026-10-22): an arm is excluded (FAIL) only for clearly broken output;
parity with M* PR is annotated, never gating.
  wer      ASR transcripts vs the reference text (Whisper English normalizer, corpus WER).
           FAIL if outputs are missing/failed, or WER > 2 x best and > 10 %.
           parity: WER <= WER(mstar_pr) + max(1.0, 2 x |WER(mstar_pr) - WER(mstar_pr_b)|).
  clip     images (or a video's middle frame) vs the prompt: CLIP ViT-L/14 cosine (x100).
           FAIL if outputs are missing/failed/undecodable/flat, or mean CLIP < best - 8 or < 15.
           parity: mean CLIP >= CLIP(mstar_pr) - 2; PSNR vs mstar_pr reported where the noise is shared.
  mcq      multiple-choice answers (first A-D letter in the reply) vs the gold letter.
           FAIL if > 10 % empty replies, or accuracy < best - 0.15. parity: accuracy >= mstar_pr - 0.05.
  text     free-form replies. FAIL if > 10 % empty. parity: mean token-set overlap with mstar_pr >= 0.5.
  cosine   float outputs (.npy, same inputs on every arm) vs the reference tag (--ref, default mstar_pr):
           FAIL if any non-finite value or mean cosine < 0.9. parity: mean cosine >= 0.99.
  frames   videos (.mp4), images, or raw uint8 RGB frames (.bin, item fields h, w): FAIL if missing/
           undecodable, flat (pixel std <= 2) or a static video. parity: PSNR vs mstar_pr (annotated).
  finite   numeric outputs (actions): FAIL if missing, empty or any non-finite value.
  cosine additionally: an item with "chunks": H must return H distinct chunks (a rollout that repeats one
           latent H times is broken output).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def load_texts(d):
    p = d / "outputs.jsonl"
    if not p.exists():
        return {}
    return {r["i"]: (r["item"], r["text"]) for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip())}


def failed_of(d):
    try:
        return json.loads((d / "DONE").read_text()).get("failed")
    except (OSError, ValueError):
        return None


def out_files(d):
    o = d / "out"
    return {int(p.stem): p for p in o.iterdir() if p.stem.isdigit()} if o.exists() else {}


def decode_image(p, item=None):
    from PIL import Image
    if p.suffix == ".bin":  # raw uint8 RGB frames [n, h, w, 3]
        h, w = int(item["h"]), int(item["w"])
        x = np.frombuffer(p.read_bytes(), dtype=np.uint8)
        frames = list(x[: len(x) // (h * w * 3) * h * w * 3].reshape(-1, h, w, 3))
        if not frames:
            raise ValueError("no frames")
        return frames[len(frames) // 2], frames
    if p.suffix == ".mp4":
        import av
        with av.open(str(p)) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        return frames[len(frames) // 2], frames
    return np.asarray(Image.open(p).convert("RGB")), None


def psnr(a, b):
    if a.shape != b.shape:
        return None
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--kind", required=True, choices=["wer", "clip", "mcq", "text", "cosine", "frames", "finite"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--ref", default="mstar_pr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [json.loads(x) for x in Path(a.items).read_text().splitlines() if x.strip()]
    root = Path(a.root)
    tags = [t for t in a.tags.split(",") if t and (root / t / a.workload).exists()]
    res = {}
    if a.kind == "wer":
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
        try:
            norm = EnglishTextNormalizer({})
        except Exception:  # noqa: BLE001
            norm = BasicTextNormalizer()
        for t in tags:
            d = root / t / a.workload
            outs = load_texts(d)
            E = N = 0
            worst = []
            for i in range(a.count):
                k, hyp = outs.get(i, (i % len(items), ""))
                ref = norm(items[k]["ref"]).split()
                h = norm(hyp).split()
                e = edits(ref, h)
                E, N = E + e, N + len(ref)
                worst.append((e / max(1, len(ref)), k, hyp[:120]))
            res[t] = {"wer": 100 * E / max(1, N), "n": len(outs), "missing": a.count - len(outs), "failed": failed_of(d),
                      "worst": sorted(worst, reverse=True)[:3]}
        best = min(r["wer"] for r in res.values() if r["n"]) if any(r["n"] for r in res.values()) else 0
        sp = abs(res["mstar_pr"]["wer"] - res["mstar_pr_b"]["wer"]) if "mstar_pr" in res and "mstar_pr_b" in res else 0
        tol = max(1.0, 2 * sp)
        for t, r in res.items():
            broken = r["missing"] > 0 or r["failed"] not in (0, None) or (r["wer"] > 2 * best and r["wer"] > 10.0)
            r["verdict"] = "FAIL (broken output)" if broken else "PASS"
            r["parity"] = "mstar_pr" in res and r["wer"] <= res["mstar_pr"]["wer"] + tol
        meta = {"best_wer": best, "same_code_spread": sp, "tolerance": tol}
    elif a.kind in ("clip", "frames"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        model = proc = None
        if a.kind == "clip":
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).cuda().eval()
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        ref_imgs = {}
        for t in [a.ref] + [x for x in tags if x != a.ref]:
            if t not in tags:
                continue
            d = root / t / a.workload
            files = out_files(d)
            scores, flat, bad, static, ps = [], 0, 0, 0, []
            for i in range(a.count):
                p = files.get(i)
                if p is None:
                    continue
                try:
                    img, frames = decode_image(p, items[i % len(items)])
                except Exception:  # noqa: BLE001
                    bad += 1
                    continue
                if float(img.std()) <= 2.0:
                    flat += 1
                if frames is not None and len(frames) > 1 and float(np.std(np.stack(frames[:16]), axis=0).mean()) < 0.5:
                    static += 1
                if t == a.ref:
                    ref_imgs[i] = img
                elif i in ref_imgs:
                    v = psnr(img, ref_imgs[i])
                    if v is not None:
                        ps.append(v)
                if model is not None:
                    k = i % len(items)
                    with torch.no_grad():
                        x = proc(text=[items[k].get("prompt", "")], images=[img], return_tensors="pt", padding=True,
                                 truncation=True).to("cuda")
                        o = model(**{kk: (vv.half() if vv.dtype == torch.float32 else vv) for kk, vv in x.items()})
                        te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
                        ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
                        scores.append(100 * float((te * ie).sum()))
            res[t] = {"n": len(files), "missing": a.count - len(files), "failed": failed_of(d), "undecodable": bad,
                      "flat": flat, "static_video": static, "clip_mean": statistics.fmean(scores) if scores else None,
                      "psnr_vs_ref_median": statistics.median(ps) if ps else None}
        best = max((r["clip_mean"] for r in res.values() if r["clip_mean"] is not None), default=None)
        for t, r in res.items():
            broken = r["missing"] > 0 or r["failed"] not in (0, None) or r["undecodable"] > 0 or r["flat"] > 0 or \
                r["static_video"] > 0
            if a.kind == "clip":
                broken = broken or r["clip_mean"] is None or r["clip_mean"] < 15 or (best and r["clip_mean"] < best - 8)
                rc = res.get(a.ref, {}).get("clip_mean")
                r["parity"] = rc is not None and r["clip_mean"] is not None and r["clip_mean"] >= rc - 2
            else:
                r["parity"] = r["psnr_vs_ref_median"] is None or r["psnr_vs_ref_median"] >= 30
            r["verdict"] = "FAIL (broken output)" if broken else "PASS"
        meta = {"best_clip": best}
    elif a.kind == "finite":
        for t in tags:
            d = root / t / a.workload
            files = out_files(d)
            bad = 0
            for i, p in files.items():
                raw = p.read_bytes()
                try:
                    v = np.asarray(json.loads(raw), dtype=np.float64).ravel()
                except ValueError:
                    v = np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<f4")
                bad += (v.size == 0) or (not np.isfinite(v).all())
            r = {"n": len(files), "missing": a.count - len(files), "failed": failed_of(d), "bad": int(bad)}
            r["verdict"] = "FAIL (broken output)" if r["missing"] or bad or r["failed"] not in (0, None) else "PASS"
            r["parity"] = None
            res[t] = r
        meta = {}
    elif a.kind in ("mcq", "text"):
        ref_out = load_texts(root / a.ref / a.workload) if a.ref in tags else {}
        for t in tags:
            d = root / t / a.workload
            outs = load_texts(d)
            empty = sum(1 for i in range(a.count) if not outs.get(i, (0, ""))[1].strip())
            r = {"n": len(outs), "missing": a.count - len(outs), "failed": failed_of(d), "empty": empty}
            if a.kind == "mcq":
                right = 0
                for i, (k, txt) in outs.items():
                    m = re.search(r"\b([A-J])\b", txt.strip()[:200])
                    right += bool(m and m.group(1) == items[k]["answer"])
                r["accuracy"] = right / max(1, a.count)
            else:
                ov = []
                for i, (k, txt) in outs.items():
                    if i in ref_out:
                        A, B = set(txt.lower().split()), set(ref_out[i][1].lower().split())
                        ov.append(len(A & B) / max(1, len(A | B)))
                r["overlap_vs_ref"] = statistics.fmean(ov) if ov else None
            res[t] = r
        best = max((r.get("accuracy", 0) for r in res.values()), default=0)
        for t, r in res.items():
            broken = r["empty"] > 0.1 * a.count or r["failed"] not in (0, None)
            if a.kind == "mcq":
                broken = broken or r["accuracy"] < best - 0.15
                r["parity"] = a.ref in res and r["accuracy"] >= res[a.ref]["accuracy"] - 0.05
            else:
                r["parity"] = r["overlap_vs_ref"] is None or r["overlap_vs_ref"] >= 0.5
            r["verdict"] = "FAIL (broken output)" if broken else "PASS"
        meta = {"best_accuracy": best}
    else:  # cosine
        def load_vec(p):
            if p.suffix == ".npy":
                return np.load(p).astype(np.float64).ravel()
            raw = p.read_bytes()
            try:
                obj = json.loads(raw)
                return np.asarray(obj, dtype=np.float64).ravel()
            except ValueError:
                return np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<f4").astype(np.float64)
        refs = {i: load_vec(p) for i, p in out_files(root / a.ref / a.workload).items()} if a.ref in tags else {}
        for t in tags:
            d = root / t / a.workload
            files = out_files(d)
            cos, nonfinite, repeated = [], 0, 0
            for i, p in files.items():
                v = load_vec(p)
                if not np.isfinite(v).all():
                    nonfinite += 1
                    continue
                H = int(items[i % len(items)].get("chunks", 1))
                if H > 1 and v.size % H == 0:
                    ch = v.reshape(H, -1)
                    repeated += all(np.array_equal(ch[0], ch[j]) for j in range(1, H))
                if i in refs and refs[i].shape == v.shape:
                    cos.append(float(v @ refs[i] / (np.linalg.norm(v) * np.linalg.norm(refs[i]) + 1e-30)))
            r = {"n": len(files), "missing": a.count - len(files), "failed": failed_of(d), "nonfinite": nonfinite,
                 "repeated_chunks": repeated,
                 "cos_mean": statistics.fmean(cos) if cos else None, "cos_min": min(cos) if cos else None}
            broken = r["missing"] > 0 or r["failed"] not in (0, None) or nonfinite > 0 or repeated > 0 or \
                (r["cos_mean"] is not None and r["cos_mean"] < 0.9)
            r["verdict"] = "FAIL (broken output)" if broken else "PASS"
            r["parity"] = r["cos_mean"] is None or r["cos_mean"] >= 0.99
            res[t] = r
        meta = {"ref": a.ref}
    Path(a.out).write_text(json.dumps({"kind": a.kind, "workload": a.workload, **meta, "verdict": res}, indent=1))
    for t, r in res.items():
        print(t, json.dumps({k: v for k, v in r.items() if k != "worst"})[:400], flush=True)


if __name__ == "__main__":
    main()
