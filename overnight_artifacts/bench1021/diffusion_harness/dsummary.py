#!/usr/bin/env python3
"""Tables from <out>/results.jsonl (tag == main): per arm and concurrency, throughput mean and
min-max over rounds (each round a fresh server process), n rounds, failed total; B=1 latency
(mean / p50 / p99 per round, averaged, with the min-max of the round means); load average.
A point counts only if every round's cell has failed == 0; otherwise it is shown as FAILED.

  python dsummary.py --out /workspace/results/bench/zimage [--tag main]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ORDER = ["mstar-pr", "mstar-main", "vllmomni", "sglang", "diffusers"]
LABEL = {"mstar-pr": "M* PR", "mstar-main": "M* main", "vllmomni": "vllm-omni", "sglang": "SGLang",
         "diffusers": "diffusers (tuned, offline batch)"}


def fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--kind", default="image", choices=["image", "video"])
    a = ap.parse_args()
    out = Path(a.out)
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("tag") == a.tag and "conc" in r]
    latest = {}
    for r in rows:  # a retry's row supersedes the failed attempt of the same cell
        latest[(r["arm"], r.get("steps"), r["conc"], r["round"])] = r
    rows = list(latest.values())
    kind = a.kind
    key_thr = "throughput_per_min" if kind == "video" else "throughput_per_s"
    unit = "videos/min" if kind == "video" else "img/s"
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r.get("steps"), r["conc"])].append(r)
    steps_main = sorted({k[1] for k in cells})
    lines = [f"# {out.name}: tag {a.tag}", ""]
    knobs = {}
    for r in rows:
        knobs.setdefault(r["arm"], set()).add(r.get("knob"))
    lines.append("Arm configs: " + "; ".join(f"{LABEL.get(k, k)} = {', '.join(sorted(map(str, v)))}"
                                              for k, v in sorted(knobs.items(), key=lambda kv: ORDER.index(kv[0])
                                                                 if kv[0] in ORDER else 99)))
    lines.append("")
    for st in steps_main:
        concs = sorted({k[2] for k in cells if k[1] == st})
        arms = [x for x in ORDER if any((x, st, c) in cells for c in concs)]
        ref = {c: None for c in concs}
        lines += [f"## Throughput ({unit}), steps={st}: mean [min-max] over rounds; n rounds; failed", "",
                  "| arm | " + " | ".join(f"c={c}" for c in concs) + " |",
                  "|---|" + "---:|" * len(concs)]
        means = {}
        for arm in arms:
            cols = []
            for c in concs:
                rs = cells.get((arm, st, c), [])
                if not rs:
                    cols.append("-")
                    continue
                fails = sum((r.get("failed") if r.get("failed") is not None else 1) for r in rs)
                vals = [r.get(key_thr) for r in rs if r.get(key_thr)]
                if fails or not vals:
                    cols.append(f"FAILED ({fails} failed, n={len(rs)})")
                    continue
                m = statistics.fmean(vals)
                means[(arm, c)] = m
                cols.append(f"{m:.3f} [{min(vals):.3f}-{max(vals):.3f}] n={len(vals)}")
            lines.append(f"| {LABEL.get(arm, arm)} | " + " | ".join(cols) + " |")
        lines.append("")
        if ("mstar-pr", concs[0]) in means:
            lines += ["M* PR throughput relative to each arm (mean / mean):", "",
                      "| vs | " + " | ".join(f"c={c}" for c in concs) + " |", "|---|" + "---:|" * len(concs)]
            for arm in arms:
                if arm == "mstar-pr":
                    continue
                cols = []
                for c in concs:
                    if ("mstar-pr", c) in means and (arm, c) in means:
                        cols.append(f"{means[('mstar-pr', c)] / means[(arm, c)]:.2f}x")
                    else:
                        cols.append("-")
                lines.append(f"| {LABEL.get(arm, arm)} | " + " | ".join(cols) + " |")
            lines.append("")
        lines += [f"## B=1 latency (s), steps={st}, c=1: mean of round means [min-max]; p50; p99 (per round, "
                  "averaged); requests per round", "", "| arm | mean | p50 | p99 | n req/round | rounds | failed |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in arms:
            rs = cells.get((arm, st, 1), [])
            if not rs:
                continue
            fails = sum((r.get("failed") if r.get("failed") is not None else 1) for r in rs)
            ms = [r["lat_mean"] for r in rs if r.get("lat_mean")]
            if not ms:
                lines.append(f"| {LABEL.get(arm, arm)} | FAILED | | | | {len(rs)} | {fails} |")
                continue
            p50 = statistics.fmean(r["lat_p50"] for r in rs if r.get("lat_p50"))
            p99 = statistics.fmean(r["lat_p99"] for r in rs if r.get("lat_p99"))
            lines.append(f"| {LABEL.get(arm, arm)} | {statistics.fmean(ms):.3f} [{min(ms):.3f}-{max(ms):.3f}] | "
                         f"{p50:.3f} | {p99:.3f} | {rs[0].get('num')} | {len(rs)} | {fails} |")
        lines.append("")
    las = [r.get("load_before") for r in rows if r.get("load_before") is not None]
    if las:
        lines.append(f"1-min load average at cell start: median {statistics.median(las):.1f}, max {max(las):.1f} "
                     f"over {len(las)} cells (192 vCPUs).")
    (out / f"summary_{a.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
