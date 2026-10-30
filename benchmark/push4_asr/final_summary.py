"""Summary tables of final_bench.py's results (~/push4/results/omni/final): ASR per model and concurrency (RTFx per
round for base / head / competitor, the head's change vs base and vs the competitor, WER, errors), long-form, realtime
latencies, and Qwen3-Omni (ab.py's table + the multi-turn per-turn table). Writes final/summary.md.

  PYTHONPATH=<head tree> python final_summary.py
"""
import json
import os
import statistics
import subprocess
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

out += ["## Qwen3-ASR realtime (0.6B), mstar only", "",
        "| sessions | partial p50 ms base / head | partial p95 ms base / head | final p50 ms base / head "
        "| WER base / head |",
        "|---|---|---|---|---|"]
for s in ("1", "8", "32"):
    row = {}
    for arm in ("base", "head"):
        runs = [load(f"{F}/rt/r{r}/{arm}/rt.json") for r in (0, 1)]
        runs = [x["sessions"][s] for x in runs if x and s in x.get("sessions", {})]
        row[arm] = runs
    def col(k, scale=1.0, nd=0, row=row):
        return " / ".join(fmt([x[k] * scale for x in row[a]], nd) for a in ("base", "head"))
    out.append(f"| {s} | {col('partial_p50_ms')} | {col('partial_p95_ms')} | {col('final_p50_ms')} | "
               f"{col('wer', 100, 2)} |")

out += ["", "## Qwen3-Omni audio->text (ab.py, tok/s / TTFT p50 ms / TPOT p50 ms per round)", "", "```"]
log = open(f"{HOME}/push4/results/omni/final_bench.log").read()
table = log[log.rfind("cell       arm"):] if "cell       arm" in log else "(ab.py table not found)"
out += [table.split("[ab] AB DONE")[0].rstrip(), "```", ""]
mt = subprocess.run([sys.executable, f"{HOME}/push4/omni/harness/mt_summary.py",
                     f"{HOME}/push4/results/omni/final/omni"], capture_output=True, text=True, check=False).stdout
out += ["### 4-turn chat, per turn", "", "```", mt.rstrip(), "```"]
open(f"{F}/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
