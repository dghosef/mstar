"""Pass/fail on the correctness stage; the benchmark runs only on PASS.

Reads what `reproduce.sh` wrote under ``--out``:

1. ``pytest_real_ckpt.log``: prompt ids (timestamps + pads), ``grid_thw`` and
   patches equal transformers' with the real tokenizer and template, and the
   mp4 decode test ran (not skipped).
2. ``vision_parity.json``: mstar's tower vs transformers' on a real video:
   fp32 cosine >= 0.9999, and bf16-vs-fp32 error no worse than 2x the
   transformers bf16-vs-fp32 yardstick.
3. ``parity/report.json``: mstar's greedy video answers teacher-forced through
   transformers. Mean argmax agreement >= 0.95 and within 0.02 of the worse of
   vLLM / SGLang, and every first disagreement a near-tie (logit gap <= 1.0,
   or no larger than the other engines' largest gap). ``mstar_a`` is gated;
   ``mstar_b`` is the same code in a second server process, reported as the
   noise floor.

  python gate.py --out ~/video_bench/results
"""
import argparse
import json
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    checks: list[tuple[str, bool, str]] = []

    log = out / "pytest_real_ckpt.log"
    text = log.read_text() if log.exists() else ""
    # the last summary line, with or without -q's missing `===` bars
    summary = (re.findall(r"^=*\s*(\d+ (?:passed|failed|error)[^=\n]*? in [\d.]+s)", text, re.M) or [""])[-1]
    # every case runs on the node (real tokenizer, FFmpeg), the mp4 decode included
    mp4_ran = "skipped" not in summary
    ok = bool(summary) and "failed" not in summary and "error" not in summary and mp4_ran
    checks.append(("prompt ids / grid_thw / patches / mp4 decode vs transformers (pytest, real tokenizer)",
                   ok, summary or "no pytest summary"))

    vp = out / "vision_parity.json"
    if vp.exists():
        r = json.loads(vp.read_text())["results"]
        fp32 = r["mstar fp32 vs HF fp32"]["cosine"]
        bf16 = r["mstar bf16 vs HF fp32"]["mean_abs"]
        yard = r["yardstick: HF bf16 vs HF fp32"]["mean_abs"]
        ok = fp32 >= 0.9999 and bf16 <= 2 * yard
        checks.append(("vision tower on video vs transformers", ok,
                       f"fp32 cosine {fp32:.6f}; bf16 mean abs {bf16:.3e} vs yardstick {yard:.3e}"))
    else:
        checks.append(("vision tower on video vs transformers", False, "no vision_parity.json"))

    rp = out / "parity" / "report.json"
    if rp.exists():
        served = json.loads(rp.read_text())["served"]

        def stats(tag):
            rows = list(served.get(tag, {}).values())
            if not rows:
                return None
            mean = sum(v["argmax_agreement"] for v in rows) / len(rows)
            gaps = [v["logit_gap_at_first"] for v in rows if v["logit_gap_at_first"] is not None]
            return mean, max(gaps, default=0.0), len(rows)

        mine = stats("mstar_a") or stats("mstar")
        control = stats("mstar_b")
        others = [s for s in (stats("vllm"), stats("sglang")) if s]
        if mine is None:
            checks.append(("teacher-forced greedy video answers", False, "no mstar answers collected"))
        else:
            floor = min((o[0] for o in others), default=1.0) - 0.02
            gap_cap = max([1.0] + [o[1] for o in others])
            ok = mine[0] >= 0.95 and mine[0] >= floor and mine[1] <= gap_cap
            detail = f"mstar agreement {mine[0]:.4f} over {mine[2]} prompts, max first-gap {mine[1]:.2f}; "
            detail += "; ".join(f"{t} {s[0]:.4f} / {s[1]:.2f}" for t, s in
                                (("mstar control (2nd process)", control), ("vllm", stats("vllm")),
                                 ("sglang", stats("sglang"))) if s)
            checks.append(("teacher-forced greedy video answers vs transformers", ok, detail))
    else:
        checks.append(("teacher-forced greedy video answers vs transformers", False, "no parity/report.json"))

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    passed = all(ok for _, ok, _ in checks)
    print("GATE", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
