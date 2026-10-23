#!/usr/bin/env python3
"""Rebuilds /workspace/results/bench/REPORT_DIFFUSION.md from what is on disk (run by queue.py after
every model, safe to run any time). Static method text: report/00_setup.txt next to this file."""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path("/workspace/results/bench")
TITLES = {"zimage": "Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo), text to image",
          "klein4b": "FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B), text to image",
          "klein9b": "FLUX.2 [klein] 9B (black-forest-labs/FLUX.2-klein-9B), text to image",
          "wan22": "Wan2.2-TI2V-5B (Wan-AI/Wan2.2-TI2V-5B-Diffusers), text to video"}
WORKLOAD = {"zimage": "1024x1024, 8 steps (8 DiT forwards), guidance 0 (Turbo, no CFG), FlowMatch Euler (checkpoint scheduler), bf16, 64 PartiPrompts",
            "klein4b": "1024x1024, 4 steps, no CFG (distilled; guidance_scale 1.0 sent, ignored), FlowMatch Euler, bf16, 64 PartiPrompts",
            "klein9b": "1024x1024, 4 steps, no CFG (distilled), FlowMatch Euler, bf16, 64 PartiPrompts",
            "wan22": "1280x704 x 81 frames (24 fps), guidance 5.0 (CFG), empty negative prompt, UniPC flow_shift 5.0 (checkpoint scheduler), bf16; 20 steps for the sweep, 50 steps for the B=1 latency row; 16 VBench prompts (gate: first 4)"}
LABEL = {"mstar-pr": "M* PR", "mstar-main": "M* main", "vllmomni": "vllm-omni", "sglang": "SGLang",
         "diffusers": "diffusers"}


def load_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def rows_latest(out, tag="main"):
    latest = {}
    for r in load_jsonl(out / "results.jsonl"):
        if r.get("tag") == tag and "conc" in r:
            latest[(r["arm"], r.get("steps"), r["conc"], r["round"])] = r
    return list(latest.values())


def means(rows, key, steps):
    acc = {}
    for r in rows:
        if r.get("steps") == steps and r.get("failed") == 0 and r.get(key):
            acc.setdefault((r["arm"], r["conc"]), []).append(r[key])
    return {k: (statistics.fmean(v), min(v), max(v), len(v)) for k, v in acc.items()}


def auto_summary(key, out, kind):
    rows = rows_latest(out)
    if not rows:
        return ["Timed rounds not run yet."]
    steps = {"wan22": 20}.get(key, {"zimage": 8}.get(key, 4))
    thr_key = "throughput_per_min" if kind == "video" else "throughput_per_s"
    mu = means(rows, thr_key, steps)
    lat = means(rows, "lat_mean", steps)
    concs = sorted({c for _, c in mu})
    lines = []
    for comp in ("mstar-main", "vllmomni", "sglang", "diffusers"):
        rs = [(c, mu[("mstar-pr", c)][0] / mu[(comp, c)][0]) for c in concs
              if ("mstar-pr", c) in mu and (comp, c) in mu]
        if not rs:
            continue
        lo, hi = min(rs, key=lambda x: x[1]), max(rs, key=lambda x: x[1])
        lat_txt = ""
        if ("mstar-pr", 1) in lat and (comp, 1) in lat:
            lat_txt = (f"; B=1 latency {lat[('mstar-pr', 1)][0]:.3f} s vs {lat[(comp, 1)][0]:.3f} s "
                       f"({lat[('mstar-pr', 1)][0] / lat[(comp, 1)][0]:.2f}x)")
        losses = [f"c={c} ({x:.2f}x)" for c, x in rs if x < 1.0]
        lines.append(f"- vs {LABEL[comp]}{' (offline batch, no serving)' if comp == 'diffusers' else ''}: "
                     f"throughput {lo[1]:.2f}x (c={lo[0]}) to {hi[1]:.2f}x (c={hi[0]}){lat_txt}. "
                     + (f"M* PR is slower at {', '.join(losses)}." if losses else "M* PR is not slower at any point."))
    missing = sorted({f"{LABEL.get(r['arm'], r['arm'])} c={r['conc']} r{r['round']}" for r in rows
                      if r.get("failed") != 0})
    if missing:
        lines.append(f"- Cells that failed (excluded): {', '.join(missing)}.")
    return lines


def section(key):
    out = RESULTS / key
    if not (out / "stages").exists():
        return []
    kind = "video" if key == "wan22" else "image"
    L = [f"## {TITLES.get(key, key)}", "", f"**Workload.** {WORKLOAD.get(key, '')}.", ""]
    done = sorted(p.stem.replace("q_", "") for p in (out / "stages").glob("q_*.done"))
    L += [f"Queue phases completed: {', '.join(done) or 'none'}.", ""]
    # gate
    g = out / "gate" / "gate.md"
    L += ["### Correctness gate (vs diffusers eager reference, same prompts and seeds)", ""]
    if g.exists():
        L += [g.read_text().strip(), ""]
        try:
            gj = json.loads((out / "gate" / "gate.json").read_text())
            for t, v in gj["verdict"].items():
                if not v["verdict"].startswith("PASS"):
                    L.append(f"- {t}: {v['verdict']}, failing checks: "
                             f"{[k for k, x in v.get('checks', {}).items() if not x]}")
            L.append("")
        except (OSError, ValueError, KeyError):
            pass
        L += ["Tags: ref_b = reference in a second process; ref_k = memory-efficient SDPA; ref_c = diffusers with "
              "the tuned arm's compile/attention; ref_seed = reference with other seeds (different noise); "
              "mstar_pr_b = a second M* PR server process (same-code control); `<engine>__<knobs>` = each "
              "competitor knob set from the tune pass. Contact sheet: "
              f"`sheet_{key}.jpg` in the results folder.", ""]
    else:
        L += ["Not run yet.", ""]
    # tune
    tj = out / "tune.json"
    if tj.exists():
        tune = json.loads(tj.read_text())
        knobs = json.loads((out / "knobs.json").read_text()) if (out / "knobs.json").exists() else {}
        tuned = json.loads((out / "tuned.json").read_text()) if (out / "tuned.json").exists() else {}
        try:
            verdict = json.loads((out / "gate" / "gate.json").read_text())["verdict"]
        except (OSError, ValueError, KeyError):
            verdict = {}
        unit = "img/s" if kind == "image" else "videos/s"
        L += ["### Competitor tuning (one server process per knob set, GPU 0 alone; score = geometric mean "
              f"throughput over the tune points, {unit})", "",
              "| engine | knob set | flags | per-point throughput | B=1 latency (s) | score | gate | chosen |",
              "|---|---|---|---|---:|---:|---|---|"]
        for eng, sets in tune.items():
            for k, v in sets.items():
                flags = " ".join(knobs.get(eng, {}).get(k, [])) or "(defaults)"
                pc = ", ".join(f"c={c}: {x:.3f}" for c, x in v.get("per_conc", {}).items() if x)
                gv = "n/a (reference impl.)" if eng == "diffusers" else verdict.get(f"{eng}__{k}", {}).get("verdict", "-")
                ch = "yes" if tuned.get(eng, {}).get("config") == k else ""
                lb = "-" if not v.get("lat_b1") else "%.3f" % v["lat_b1"]
                sc = "failed" if not v.get("score") else "%.3f" % v["score"]
                L.append(f"| {LABEL.get(eng, eng)} | {k} | `{flags}` | {pc or 'failed'} | {lb} | {sc} | {gv} | {ch} |")
        L.append("")
    # commands
    cmds = sorted((out / "servers").glob("main_*_r1.cmd"))
    if cmds:
        L += ["### Exact server commands (round 1)", "", "```"]
        for c in cmds:
            L.append(c.read_text().splitlines()[0])
        L += ["```", ""]
    vd = out / "versions"
    if vd.exists():
        L += ["Versions: " + "; ".join(f"{p.stem}: {json.dumps(json.loads(p.read_text()))[:300]}"
                                         for p in sorted(vd.glob("*.json"))), ""]
    # tables
    s = out / "summary_main.md"
    L += ["### Timed results (3 rounds, fresh server per round, rotated order)", ""]
    L += [s.read_text().split("\n", 2)[-1].strip(), ""] if s.exists() else ["Not run yet.", ""]
    # profile
    for pf in sorted((out / "profile").glob("phases_c*.txt")) if (out / "profile").exists() else []:
        txt = pf.read_text()
        verdict = profile_verdict(txt)
        L += [f"### Profile ({pf.stem}, MSTAR_PHASE_TIMING=100, benchmark/worker_phases): {verdict}", "",
              "<details><summary>phase tables</summary>", "", "```", txt.strip()[:6000], "```", "", "</details>", ""]
    # notes / summary
    L += ["### Summary (generated from the tables above)", ""] + auto_summary(key, out, kind) + [""]
    n = out / "notes.txt"
    if n.exists():
        L += ["<details><summary>notes.txt (startup times, health checks, decisions)</summary>", "", "```",
              n.read_text().strip()[:8000], "```", "", "</details>", ""]
    return L


def profile_verdict(txt):
    """event_sync is the only real GPU wait in the worker loop: a sizeable share of iter_total means
    a GPU-bound component; near zero means CPU-bound."""
    ev = [float(x) for x in re.findall(r"\bevent_sync\s+([\d.]+)", txt)]
    it = [float(x) for x in re.findall(r"\biter_total\s+([\d.]+)", txt)]
    if not ev or not it:
        return "verdict unavailable (no event_sync / iter_total rows)"
    share = max(e / i for e, i in zip(ev, it) if i > 0) if any(it) else 0
    kind = "GPU-bound (worker waits on the GPU most of each step)" if share > 0.5 else \
        ("mixed" if share > 0.1 else "CPU-bound")
    return f"{kind}; event_sync / iter_total up to {share:.0%}"


def main():
    setup = (HERE / "report" / "00_setup.txt").read_text()
    L = ["# Diffusion lane: M* vs vllm-omni, SGLang diffusion and diffusers (node2, 8x H200)", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by report_diffusion.py from "
         "/workspace/results/bench/<model>/ on node2._", "", setup.strip(), ""]
    for key in ("zimage", "klein4b", "wan22", "klein9b"):
        L += section(key)
    cav = HERE / "report" / "caveats.txt"
    if cav.exists():
        L += ["## Caveats", "", cav.read_text().strip(), ""]
    dec = load_jsonl(RESULTS / "decisions.jsonl")
    skipped = [d for d in dec if d["kind"] in ("model_skipped", "deadline_skip", "deadline_skip_phase",
                                                  "phase_abandoned", "pass_failed", "collect_failed")]
    L += ["## Not covered / failed", ""]
    L += [f"- {d['time']} {d['model']}: {d['kind']} {json.dumps({k: v for k, v in d.items() if k not in ('time', 'model', 'kind')})}"
          for d in skipped] or ["- (nothing recorded)"]
    L += ["- Multi-GPU rows (Wan2.2 CFG split on 2 GPUs, DP2) were not run: out of the night's budget.", ""]
    (RESULTS / "REPORT_DIFFUSION.md").write_text("\n".join(L) + "\n")
    print(RESULTS / "REPORT_DIFFUSION.md")


if __name__ == "__main__":
    main()
