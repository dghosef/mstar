#!/usr/bin/env python3
"""Correctness gate for the diffusion lane: every arm's outputs vs the diffusers reference,
same prompts and seeds (prompt i, seed seed_base + i).

Directories under --root, one per tag, each holding <i>.png (images) or <i>.mp4 / <i>.npy (video):
  ref_a     diffusers eager, default attention dispatch (torch SDPA), process 1   <- the reference
  ref_b     same settings, a second process on another GPU                       (cross-process floor)
  ref_k     diffusers eager, memory-efficient SDPA instead of the default kernel  (kernel yardstick)
  ref_c     diffusers with the tuned arm's torch.compile                          (compile yardstick)
  ref_seed  ref_a's settings with seed + 500                                      (what different noise looks like)
  <arm>     each served arm's outputs (mstar_pr, mstar_main, vllmomni, sglang), and mstar_pr_b
            (a second mstar-pr server process, the same-code control)

Metrics per item: PSNR (uint8 RGB, peak 255; identical -> 100 dB), SSIM (skimage, channel_axis),
CLIP score (100 * cos(image, text), openai/clip-vit-large-patch14, clamped at 0). Video: PSNR over
all frames of the arm's decoded mp4 vs the reference's raw uint8 frames (never through a codec),
the worst frame's PSNR, SSIM on every 10th frame, CLIP on frames 0 / mid / last. The yardsticks
are decoded from their own mp4s too, so they include one codec pass like the served arms.

Pass criteria, fixed before the run (2026-10-22, before any arm was collected):
  Y = for each statistic, the worst value among the valid-choice yardsticks ref_b, ref_k, ref_c
      (vs ref_a).
  An arm is "seed-matched" if its median PSNR vs ref_a exceeds ref_seed's median by >= 10 dB.
  Seed-matched arm PASSES if all hold:
    median PSNR >= Y.median_psnr - 2 dB;  p10 PSNR >= Y.p10_psnr - 3 dB;
    median SSIM >= Y.median_ssim - 0.03;  mean CLIP >= ref_a mean CLIP - 1.0;
    every output valid (no failed request, right size/frames, not flat).
  Not seed-matched (its noise is drawn differently, e.g. a CUDA generator): PASS (distribution)
    if mean CLIP >= ref_a mean CLIP - 1.0 and every output valid; labelled as a distribution-level
    check only (pixel parity cannot be tested).
  Nothing is loosened after seeing results; a FAIL is reported with the numbers.

AMENDMENT (2026-10-22 06:50 UTC, after observing that independent engines are not pixel-matched to the
diffusers reference: vllm-omni and SGLang landed at 23.8-24.3 dB median PSNR on Z-Image, against a
different-seed control of 10.9 dB and equal CLIP). Two tiers; the original verdict stays in the table:
  M* arms (tags mstar_*): unchanged (the criteria above: parity with the reference implementation).
  Competitor arms (every other non-yardstick tag): "same image for the same seed" check. PASS if every
    output is valid, mean CLIP >= ref_a mean CLIP - 0.5, and median PSNR vs ref_a >=
    (ref_seed median + Y median) / 2. Median SSIM is reported.

GATE RULE (2026-10-22 ~07:00 UTC, user direction; supersedes the two above as the gating rule): a sanity
gate. Parity is reported, not gating. An arm (M* included) is excluded only if its output is clearly broken:
invalid/blank outputs, mean CLIP more than 2 points below ref_a, or median PSNR <= ref_seed median + 3 dB
(a different image). The strict-parity result (the original criteria for M*, the amended same-image check for
competitors) is shown as "parity: yes/no (annotated)".
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else min(100.0, 10 * math.log10(255.0 ** 2 / mse))


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def read_video(p: Path) -> np.ndarray:
    import av
    with av.open(str(p)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def load_item(d: Path, i: int, kind: str, raw: bool):
    if kind == "image":
        p = d / f"{i:03d}.png"
        return np.asarray(Image.open(p).convert("RGB")) if p.exists() else None
    if raw and (d / f"{i:03d}.npy").exists():
        return np.load(d / f"{i:03d}.npy")
    p = d / f"{i:03d}.mp4"
    return read_video(p) if p.exists() else None


class Clip:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device

    def score(self, images: list[np.ndarray], text: str) -> float:
        import torch
        with torch.no_grad():
            inp = self.p(text=[text], images=[Image.fromarray(x) for x in images], return_tensors="pt",
                         padding=True, truncation=True, max_length=77).to(self.device)
            out = self.m(**inp)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            return float(max(0.0, 100 * (ie @ te.T).mean().item()))


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    s = sorted(xs)
    return {"n": len(s), "mean": float(np.mean(s)), "median": float(np.median(s)),
            "p10": float(np.percentile(s, 10)), "min": float(s[0]), "max": float(s[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--kind", required=True, choices=["image", "video"])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--arms", required=True, help="comma list of served-arm tags")
    ap.add_argument("--failed-json", default=None, help="{tag: failed_count} from collection")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.root)
    prompts = [p for p in Path(a.prompts).read_text().splitlines() if p.strip()]
    clip = Clip(a.device)
    failed = json.loads(Path(a.failed_json).read_text()) if a.failed_json else {}
    tags = ["ref_b", "ref_k", "ref_c", "ref_seed"] + [t for t in a.arms.split(",") if t]
    tags = [t for t in tags if (root / t).exists()]
    per = {t: {"psnr": [], "psnr_worst_frame": [], "ssim": [], "clip": [], "missing": 0} for t in ["ref_a"] + tags}

    for i in range(a.count):
        ref = load_item(root / "ref_a", i, a.kind, raw=True)
        if ref is None:
            raise SystemExit(f"reference item {i} missing")
        text = prompts[i % len(prompts)]
        frames_for_clip = (lambda v: [v] if a.kind == "image" else [v[0], v[len(v) // 2], v[-1]])
        per["ref_a"]["clip"].append(clip.score(frames_for_clip(ref), text))
        for t in tags:
            x = load_item(root / t, i, a.kind, raw=False)
            if x is None or x.shape != ref.shape:
                per[t]["missing"] += 1
                continue
            if a.kind == "image":
                per[t]["psnr"].append(psnr(x, ref))
                per[t]["ssim"].append(ssim(x, ref))
            else:
                per[t]["psnr"].append(psnr(x, ref))
                per[t]["psnr_worst_frame"].append(min(psnr(x[f], ref[f]) for f in range(len(ref))))
                per[t]["ssim"].append(float(np.mean([ssim(x[f], ref[f]) for f in range(0, len(ref), 10)])))
            per[t]["clip"].append(clip.score(frames_for_clip(x), text))
        print(f"item {i} done", flush=True)

    summ = {t: {k: stats(v) for k, v in d.items() if k != "missing"} | {"missing": d["missing"]}
            for t, d in per.items()}
    ys = [summ[t] for t in ("ref_b", "ref_k", "ref_c") if t in summ and summ[t]["psnr"]]
    Y = {"median_psnr": min(s["psnr"]["median"] for s in ys), "p10_psnr": min(s["psnr"]["p10"] for s in ys),
         "median_ssim": min(s["ssim"]["median"] for s in ys)}
    ref_clip = summ["ref_a"]["clip"]["mean"]
    seed_med = summ["ref_seed"]["psnr"]["median"] if "ref_seed" in summ and summ["ref_seed"]["psnr"] else 0.0
    verdict = {}
    same_image = (seed_med + Y["median_psnr"]) / 2
    for t in [x for x in tags if not x.startswith("ref_")]:
        s = summ[t]
        sanity = {"valid_outputs": s["missing"] == 0 and int(failed.get(t, 0) or 0) == 0,
                  "clip_within_2": bool(s["clip"] and s["clip"]["mean"] >= ref_clip - 2.0),
                  "same_image": bool(s["psnr"] and s["psnr"]["median"] > seed_med + 3.0)}
        valid = s["missing"] == 0 and int(failed.get(t, 0)) == 0
        clip_ok = s["clip"] and s["clip"]["mean"] >= ref_clip - 1.0
        if not s["psnr"]:
            verdict[t] = {"verdict": "FAIL (broken output)", "why": "no comparable outputs", "parity": False}
            continue
        matched = s["psnr"]["median"] >= seed_med + 10.0
        checks = {"valid_outputs": valid, "clip_mean": bool(clip_ok)}
        if matched:
            checks.update(median_psnr=s["psnr"]["median"] >= Y["median_psnr"] - 2.0,
                          p10_psnr=s["psnr"]["p10"] >= Y["p10_psnr"] - 3.0,
                          median_ssim=s["ssim"]["median"] >= Y["median_ssim"] - 0.03)
        ok = all(checks.values())
        original = ("PASS" if ok else "FAIL") + ("" if matched else " (distribution-level only)")
        if t.startswith("mstar_"):
            verdict[t] = {"seed_matched": matched, "checks": checks, "verdict_original": original,
                          "tier": "parity (M*)", "parity": original.startswith("PASS")}
        else:
            c2 = {"valid_outputs": valid, "clip_mean": bool(s["clip"] and s["clip"]["mean"] >= ref_clip - 0.5),
                  "same_image_psnr": s["psnr"]["median"] >= same_image}
            verdict[t] = {"seed_matched": matched, "checks": c2, "checks_original": checks,
                          "verdict_original": original, "tier": "same-image (competitor, amended)",
                          "parity": all(c2.values())}
        verdict[t]["sanity"] = sanity
        verdict[t]["verdict"] = "PASS" if all(sanity.values()) else "FAIL (broken output)"
    report = {"yardstick": Y, "ref_clip_mean": ref_clip, "ref_seed_median_psnr": seed_med,
              "same_image_psnr_threshold": same_image, "summary": summ,
              "verdict": verdict, "per_item": per}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(report, indent=1))
    lines = [f"Yardstick Y (worst of ref_b/ref_k/ref_c vs ref_a): median PSNR {Y['median_psnr']:.2f} dB, "
             f"p10 PSNR {Y['p10_psnr']:.2f} dB, median SSIM {Y['median_ssim']:.4f}; ref_a CLIP {ref_clip:.2f}; "
             f"different-seed control median PSNR {seed_med:.2f} dB; competitor same-image threshold "
             f"{same_image:.2f} dB (amended gate, see dgate.py)", "",
             "| tag | median PSNR | p10 PSNR | min PSNR | median SSIM | mean CLIP | missing | sanity gate | parity (annotated) | parity tier | original pre-registered verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for t in tags:
        s = summ[t]
        if not s["psnr"]:
            v = verdict.get(t, {})
            lines.append(f"| {t} | - | - | - | - | - | {s['missing']} | {v.get('verdict', '')} | no | "
                         f"{v.get('tier', '')} | {v.get('verdict_original', '')} |")
            continue
        lines.append(f"| {t} | {s['psnr']['median']:.2f} | {s['psnr']['p10']:.2f} | {s['psnr']['min']:.2f} | "
                     f"{s['ssim']['median']:.4f} | {s['clip']['mean']:.2f} | {s['missing']} | "
                     f"{verdict.get(t, {}).get('verdict', 'yardstick')} | "
                     f"{'-' if t.startswith('ref_') else ('yes' if verdict.get(t, {}).get('parity') else 'no (annotated)')} | "
                     f"{verdict.get(t, {}).get('tier', '-')} | {verdict.get(t, {}).get('verdict_original', '-')} |")
    lines.append(f"| ref_a | - | - | - | - | {ref_clip:.2f} | 0 | reference | - | - | - |")
    (out / "gate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
