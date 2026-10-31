"""Summary tables of final_bench.py's results (~/push4/results/omni/final): ASR per model and concurrency (RTFx per
round for base / head / competitor, the head's change vs base and vs the competitor, WER, errors), long-form, realtime
latencies, and Qwen3-Omni (ab.py's table + the multi-turn per-turn table). Writes final/summary.md.

  PYTHONPATH=<head tree> python final_summary.py
"""
import json
import os
import statistics
import sys

from benchmark.asr_bench import make_normalizer, word_error_rate

HOME = os.path.expanduser("~")
F = f"{HOME}/push4/results/omni/final"
sys.path.insert(0, f"{HOME}/push4/asr/harness")
from final_bench import ASR  # noqa: E402

REFS_DIR = f"{HOME}/push4/asr/data/test_clean_200"
refs = dict(line.split("\t", 1) for line in open(f"{REFS_DIR}/references.tsv").read().splitlines() if line.strip())
norm = make_normalizer("whisper")
CONCS = ["1", "8", "32", "64"]


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def served(arm, engine, key):
    """{conc: (rtfx per round, wer per round, errors)} and long-form (rtfx, wer) per round."""
    cells, lf = {c: ([], [], 0) for c in CONCS}, []
    for r in (0, 1):
        b = load(f"{F}/asr/r{r}/{arm}/{engine}_{key}/bench.json")
        if not b:
            continue
        for c in CONCS:
            x = b["concurrency"].get(c)
            if x:
                rt, we, er = cells[c]
                cells[c] = (rt + [x["rtfx_median"]], we + [x["wer"]], er + x["errors"])
        if b.get("long_form") and b["long_form"].get("rtfx_median"):
            lf.append((b["long_form"]["rtfx_median"], b["long_form"]["wer"]))
    return cells, lf


def nemo(key):
    cells = {c: ([], [], 0) for c in CONCS}
    for r in (0, 1):
        n = load(f"{F}/asr/r{r}/nemo/{key}/nemo.json")
        for p in (n or {}).get("points", []):
            c = str(p["batch"])
            uids = list(refs)
            w = word_error_rate([refs[u] for u in uids], [p["hyps"].get(u, "") for u in uids], norm)
            rt, we, er = cells[c]
            cells[c] = (rt + [p["rtfx"]], we + [w], er)
    return cells


def fmt(xs, nd=0):
    return " / ".join(f"{x:.{nd}f}" for x in xs) if xs else "-"


def pct(a, b):
    return f"{(statistics.mean(a) / statistics.mean(b) - 1) * 100:+.0f}%" if a and b else "-"


out = ["# Final bench: ASR + Qwen3-Omni audio->text (push4a H100, GPUs 2,3)", "",
       "Arms: base d48d4da93, head 65ee34618 (Canary: 65ee34618 + ccb3c770e), competitor vLLM 0.32 / NeMo 3.0.0 "
       "(offline batched transcribe, batch = c) / SGLang-Omni 0.1.9. 2 rounds (r0 / r1, arm order reversed in r1), "
       "fresh server per arm and round, 16-request warmup excluded, each cell the median of 2 repeats. WER: whisper "
       "normalizer, 200 LibriSpeech test-clean utterances.", ""]
for label, model, comp, comp_key in ASR:
    base, base_lf = served("base", "mstar", model)
    head, head_lf = served("head", "mstar", model)
    other = nemo(comp_key) if comp == "nemo" else served("vllm", "vllm", comp_key)[0]
    out += [f"## {label} (competitor: {comp})", "",
            "| c | base RTFx | head RTFx | comp RTFx | head vs base | head vs comp | WER base / head / comp | errors |",
            "|---|---|---|---|---|---|---|---|"]
    for c in CONCS:
        b, h, o = base[c], head[c], other[c]
        wer = " / ".join(f"{statistics.mean(x[1]) * 100:.2f}%" if x[1] else "-" for x in (b, h, o))
        out.append(f"| {c} | {fmt(b[0])} | {fmt(h[0])} | {fmt(o[0])} | {pct(h[0], b[0])} | {pct(h[0], o[0])} | "
                   f"{wer} | {b[2]}/{h[2]}/{o[2]} |")
    lf = "; ".join(f"{name} RTFx {fmt([x[0] for x in v])} WER {fmt([x[1] * 100 for x in v], 2)}%"
                   for name, v in (("base", base_lf), ("head", head_lf)) if v)
    out += ["", f"Long-form 605 s, c1: {lf or '-'}", ""]

out += ["## Qwen3-ASR realtime (0.6B), mstar only (rt2: 50 utterances at real time, r0 / r1)", "",
        "First partial: audio-to-partial delay of each session's first partial, and its time since the session's "
        "first audio frame. Final: commit to completed transcript.", "",
        "| sessions | first partial p50 / p99 ms, base | head | since first frame p50 ms, base / head "
        "| final p50 / p99 ms, base | head | WER base / head | errors base / head |",
        "|---|---|---|---|---|---|---|---|"]
for sess in ("1", "8", "32"):
    runs = {arm: [x["sessions"][sess] for x in (load(f"{F}/rt2/r{r}/{arm}/rt.json") for r in (0, 1))
                  if x and sess in x.get("sessions", {})] for arm in ("base", "head")}

    def pp(arm, k50, k99, runs=runs):
        return " ; ".join(f"{x[k50]:.0f} / {x[k99]:.0f}" for x in runs[arm]) or "-"

    out.append(f"| {sess} | {pp('base', 'first_partial_p50_ms', 'first_partial_p99_ms')} "
               f"| {pp('head', 'first_partial_p50_ms', 'first_partial_p99_ms')} "
               f"| {fmt([x['first_partial_since_start_p50_ms'] for x in runs['base']])} / "
               f"{fmt([x['first_partial_since_start_p50_ms'] for x in runs['head']])} "
               f"| {pp('base', 'final_p50_ms', 'final_p99_ms')} | {pp('head', 'final_p50_ms', 'final_p99_ms')} "
               f"| {fmt([x['wer'] * 100 for x in runs['base']], 2)} / {fmt([x['wer'] * 100 for x in runs['head']], 2)} "
               f"| {sum(x['errors'] for x in runs['base'])} / {sum(x['errors'] for x in runs['head'])} |")

ARMS = ("base", "head", "sgomni")
rows = [json.loads(line) for r in (0, 1) for line in open(f"{F}/omni/r{r}/rows.jsonl") if line.strip()] \
    if os.path.exists(f"{F}/omni/r0/rows.jsonl") else []
out += ["", "## Qwen3-Omni audio->text (ab.py, 2 rounds rotated; values r0 / r1)", "",
        "Short clips: the Oct-21 a2t set (2-30 s); long: ~67 s and ~187 s LibriSpeech joins, 32 distinct clips per "
        "cell. FLAG = head's TTFT (mean of the rounds) worse than SGLang-Omni's.", "",
        "| cell | tok/s base | head | SGLang-Omni | TTFT p50 ms base | head | SGLang-Omni "
        "| TTFT p99 ms base | head | SGLang-Omni | failed | flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
cells = sorted({(x["kind"], x["conc"]) for x in rows if x["kind"] != "mta2t"},
               key=lambda kc: (["a2t", "a2tlong60", "a2tlong180"].index(kc[0]), kc[1]))
for kind, conc in cells:
    by = {a: [x for x in rows if x["kind"] == kind and x["conc"] == conc and x["arm"] == a] for a in ARMS}
    col = {a: {k: [x[k] * (1000 if k.startswith("ttft") else 1) for x in by[a] if x.get(k) is not None]
               for k in ("output_tok_s", "ttft_p50", "ttft_p99")} for a in ARMS}
    flags = [name for name, k in (("p50", "ttft_p50"), ("p99", "ttft_p99"))
             if col["head"][k] and col["sgomni"][k]
             and statistics.mean(col["head"][k]) > statistics.mean(col["sgomni"][k])]
    failed = "/".join(str(sum(x.get("failed") or 0 for x in by[a])) for a in ARMS)
    out.append(f"| {kind} c{conc} | " + " | ".join(fmt(col[a]["output_tok_s"]) for a in ARMS) + " | "
               + " | ".join(fmt(col[a]["ttft_p50"]) for a in ARMS) + " | "
               + " | ".join(fmt(col[a]["ttft_p99"]) for a in ARMS)
               + f" | {failed} | {'FLAG ' + ','.join(flags) if flags else ''} |")

out += ["", "### 4-turn chat (omni_mt rerun; per turn, values r0 / r1)", "",
        "| cell | turn | tok/s base | head | SGLang-Omni | TTFT p50 ms base | head | SGLang-Omni "
        "| TTFT p99 ms base | head | SGLang-Omni | failed | flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for conc in (1, 16):
    turns = {a: [load(f"{F}/omni_mt/r{r}/{a}/mta2t_c{conc}/mt.json") for r in (0, 1)] for a in ARMS}
    for t in range(4):
        col = {a: {k: [m["summary"][t][k] for m in turns[a] if m and len(m["summary"]) > t]
                   for k in ("out_tok_s", "ttft_p50_ms", "ttft_p99_ms", "failed")} for a in ARMS}
        flags = [name for name, k in (("p50", "ttft_p50_ms"), ("p99", "ttft_p99_ms"))
                 if col["head"][k] and col["sgomni"][k]
                 and statistics.mean(col["head"][k]) > statistics.mean(col["sgomni"][k])]
        out.append(f"| c{conc} | {t + 1} | " + " | ".join(fmt(col[a]["out_tok_s"]) for a in ARMS) + " | "
                   + " | ".join(fmt(col[a]["ttft_p50_ms"]) for a in ARMS) + " | "
                   + " | ".join(fmt(col[a]["ttft_p99_ms"]) for a in ARMS)
                   + f" | {'/'.join(str(sum(col[a]['failed'])) for a in ARMS)} "
                   f"| {'FLAG ' + ','.join(flags) if flags else ''} |")
open(f"{F}/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
