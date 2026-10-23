"""Render REPORT.md from the rsynced results (runs on the Mac, stdlib only).

  python3 report.py --results ~/runs/bench1021/results --out ~/runs/bench1021/REPORT.md --status "..."
Per model: notes.txt (hand-written status, sha, caveats), summary.md (bench.py summarize),
competitor tuning (results.jsonl tag=tune, tuned.json, tune_failed.jsonl), correctness
(gate.txt, gsm8k_*.json, parity/report.json), client cross-check, profile, startup times.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["q35_0.8b", "q35_9b", "q35_9b_tp2", "q35_27b_tp4", "q35_0.8b_vl", "q35_9b_vl", "qwen3_omni"]


def rows(d: Path):
    p = d / "results.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def jl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def tuning(d: Path) -> list[str]:
    rs = [r for r in rows(d) if r["tag"] == "tune"]
    if not rs:
        return []
    tuned = json.loads((d / "tuned.json").read_text()) if (d / "tuned.json").exists() else {}
    failed = jl(d / "tune_failed.jsonl")
    cmds = {}
    for f in sorted((d / "servers").glob("tune_*.cmd")):
        cmds[f.stem] = f.read_text().splitlines()[0]
    L = ["#### Competitor tuning", "",
         "Short sweep, random 1024/256, c = 1 (8 req), 32 (96 req), 128 (256 req); one server per knob set; "
         "chosen = best geometric-mean output tok/s over the three points.", "",
         "| engine | knob set | c=1 tok/s | c=32 tok/s | c=128 tok/s | geomean | chosen |", "|---|---|---:|---:|---:|---:|:---:|"]
    for eng in ("vllm", "sglang", "mstar-pr", "mstar-pr-rf"):
        names = []
        for r in rs:
            if r["arm"] == eng and r["config"] not in names:
                names.append(r["config"])
        for f in failed:
            if f["engine"] == eng and f["knob"] not in names:
                names.append(f["knob"])
        for n in names:
            cell = {r["conc"]: r for r in rs if r["arm"] == eng and r["config"] == n}
            vals = []
            out = []
            for c in (1, 32, 128):
                r = cell.get(c)
                if r and r.get("failed") == 0 and r.get("output_tok_s"):
                    out.append(f"{r['output_tok_s']:,.0f}")
                    vals.append(r["output_tok_s"])
                else:
                    out.append("fail" if r else ("did not start" if any(f["knob"] == n and f["engine"] == eng
                                                                         for f in failed) else "-"))
            gm = f"{math.exp(statistics.mean(math.log(v) for v in vals)):,.0f}" if len(vals) == 3 else "-"
            mark = "✓" if tuned.get(eng, {}).get("config") == n else ""
            if eng.startswith("mstar"):
                mark = "✓" if (tuned.get("mstar_frontend") == "rust") == eng.endswith("-rf") else ""
            L.append(f"| {eng} | {n} | " + " | ".join(out) + f" | {gm} | {mark} |")
    L.append("")
    for eng in ("vllm", "sglang"):
        if eng in tuned:
            t = tuned[eng]
            L.append(f"- {eng} final extra flags: `{' '.join(t['args']) or '(none beyond the matched base)'}`")
    if "mstar_frontend" in tuned:
        L.append(f"- M* HTTP frontend: **{tuned['mstar_frontend']}** for every M* arm of this model "
                 f"({tuned.get('mstar_frontend_reason', '')})")
    L.append("")
    return L


def correctness(d: Path) -> list[str]:
    L = []
    gate = d / "gate.txt"
    if gate.exists():
        L += ["#### Correctness gate", "", "```", gate.read_text().strip(), "```", ""]
    rep = d / "parity" / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        L += ["| run | mean argmax agreement | prompts fully argmax | min confident agreement | gaps at first disagreement | ranks |",
              "|---|---:|---:|---:|---|---|"]
        for tag, rowsd in r["served"].items():
            v = list(rowsd.values())
            mean = sum(x["argmax_agreement"] for x in v) / len(v)
            full = sum(x["first_disagreement"] is None for x in v)
            conf = [x["confident_agreement"] for x in v if x.get("confident_agreement") is not None]
            gaps = ", ".join(f"{x['logit_gap_at_first']:.2f}" for x in v if x["logit_gap_at_first"] is not None)
            ranks = ", ".join(str(x["rank_at_first"]) for x in v if x["rank_at_first"] is not None)
            L.append(f"| {tag} | {mean:.4f} | {full}/{len(v)} | {min(conf) if conf else '-'} | {gaps or '-'} | {ranks or '-'} |")
        L.append("")
    if (d / "gate_summary.txt").exists():
        L += ["Gate verdicts (rule per model kind in harness/benchq.py):", "", "```",
              (d / "gate_summary.txt").read_text().strip(), "```", ""]
    for pref, label, key in (("mmmu", "MMMU-200 accuracy", "accuracy"), ("wer_a2t", "LibriSpeech test-clean A2T WER", "wer"),
                             ("wer_t2s", "T2S Whisper WER vs own text", "wer")):
        fs = sorted(d.glob(f"{pref}_*.json"))
        if fs:
            L += [f"| {label} | value | n |", "|---|---:|---:|"]
            for f in fs:
                j = json.loads(f.read_text())
                v = j.get(key)
                L.append(f"| {f.stem[len(pref) + 1:]} | {v:.4f} | {j.get('n')} |" if v is not None else
                         f"| {f.stem[len(pref) + 1:]} | - | {j.get('n')} |")
            L.append("")
    gs = sorted(d.glob("gsm8k_*.json"))
    if gs:
        L += ["| GSM8K (250 q, 5-shot, greedy) | accuracy | errors |", "|---|---:|---:|"]
        for g in gs:
            j = json.loads(g.read_text())
            L.append(f"| {g.stem[6:]} | {j['accuracy']:.3f} | {j['errors']} |")
        L.append("")
    if (d / "gallery").exists():
        L += [f"Gallery: `results/{d.name}/gallery/gallery.html`", ""]
    return L


def xcheck(d: Path) -> list[str]:
    xr = [r for r in rows(d) if r["tag"] == "xcheck"]
    if not xr:
        return []
    L = ["#### Client cross-check (random 1024/256, c=32, 160 requests, same server)", "",
         "| server | our client tok/s | our TTFT p50 ms | our TPOT p50 ms | vllm bench serve tok/s | TTFT p50 | TPOT p50 | sglang.bench_serving tok/s | TTFT p50 | TPOT p50 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in xr:
        arm = r["arm"]
        vb = d / "xcheck" / f"vllm_bench_{arm}.json"
        sb = d / "xcheck" / f"sglang_bench_{arm}.jsonl"
        v = json.loads(vb.read_text()) if vb.exists() else {}
        s = jl(sb)[-1] if sb.exists() and jl(sb) else {}
        def f(x, k, scale=1.0, fmt="{:,.0f}"):
            return fmt.format(x[k] * scale) if x.get(k) is not None else "-"
        L.append(f"| {arm} | {r.get('output_tok_s') or 0:,.0f} | {(r.get('ttft_p50') or 0) * 1e3:.1f} | "
                 f"{(r.get('tpot_p50') or 0) * 1e3:.2f} | {f(v, 'output_throughput')} | {f(v, 'median_ttft_ms', fmt='{:.1f}')} | "
                 f"{f(v, 'median_tpot_ms', fmt='{:.2f}')} | {f(s, 'output_throughput')} | "
                 f"{f(s, 'median_ttft_ms', fmt='{:.1f}')} | {f(s, 'median_tpot_ms', fmt='{:.2f}')} |")
    L.append("")
    return L


def startup(d: Path) -> list[str]:
    st = jl(d / "startup.jsonl")
    if not st:
        return []
    by = {}
    for s in st:
        if s["label"].startswith("main_"):
            by.setdefault(s["arm"], []).append(s["seconds"])
    if not by:
        return []
    return ["Server startup to healthy (main rounds), seconds: " +
            ", ".join(f"{a} {statistics.mean(v):.0f} ({min(v):.0f}-{max(v):.0f})" for a, v in by.items()), ""]


def profile(d: Path) -> list[str]:
    L = []
    for p in sorted(d.glob("phases_*.txt")):
        txt = p.read_text().strip()
        L += [f"#### Worker phases ({p.stem[7:]}, MSTAR_PHASE_TIMING=100)", "", "<details><summary>table</summary>", "",
              "```", txt[-12000:], "```", "", "</details>", ""]
    v = d / "profile_verdict.txt"
    if v.exists():
        L += [v.read_text().strip(), ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path.home() / "runs/bench1021/results")
    ap.add_argument("--out", type=Path, default=Path.home() / "runs/bench1021/REPORT.md")
    ap.add_argument("--status", default="in progress")
    a = ap.parse_args()
    head = (HERE / "report_header.txt").read_text()
    hw = a.results / "hardware.txt"
    notes = HERE / "hardware_notes.txt"
    hw_txt = (notes.read_text().strip() + "\n\n" if notes.exists() else "") + \
        ("```\n" + hw.read_text().strip() + "\n```" if hw.exists() else "")
    head = head.replace("{STATUS}", a.status).replace("{HARDWARE}", hw_txt or "(pending)")
    L = [head, "## Results", ""]
    models = [m for m in ORDER if (a.results / m).exists()]
    for m in models:
        d = a.results / m
        L += [f"## {m}", ""]
        if (d / "notes.txt").exists():
            L += ["```", (d / "notes.txt").read_text().strip(), "```", ""]
        if False:
            L += [(d / "notes.txt").read_text().strip(), ""]
        if (d / "summary.md").exists():
            L += [(d / "summary.md").read_text().strip(), ""]
        L += startup(d) + correctness(d) + tuning(d) + xcheck(d) + profile(d)
    g = HERE / "caveats.txt"
    L += ["## Open caveats", "", g.read_text().strip() if g.exists() else "(none recorded yet)", ""]
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
