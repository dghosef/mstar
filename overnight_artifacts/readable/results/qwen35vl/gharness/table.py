"""Tables from pulled results: python3 table.py <results_root> r2 r8 ... (run dirs)."""
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])


def runner_metrics(log: Path):
    if not log.exists():
        return None
    t = log.read_text(errors="replace")
    tok = re.search(r"Throughput: ([\d.]+) text tok/s", t)
    ttft = re.search(r"TTFT \(text\)\s+: mean=([\d.]+)s\s+p50=([\d.]+)s\s+p95=([\d.]+)s", t)
    fails = re.search(r"Requests : (\d+)/(\d+) succeeded", t)
    return {
        "tok_s": float(tok.group(1)) if tok else None,
        "ttft_p50_ms": 1e3 * float(ttft.group(2)) if ttft else None,
        "ttft_p95_ms": 1e3 * float(ttft.group(3)) if ttft else None,
        "ok": fails.group(1) == fails.group(2) if fails else None,
    }


def qcheck(d: Path):
    f = d / "results.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())["summary"]


def asr(d: Path):
    js = list(d.glob("asr-*.json"))
    if not js:
        return None
    data = json.loads(js[0].read_text())
    reps = data.get("repeats") or data.get("results") or []
    rows = []
    for r in reps:
        m = r.get("metrics", r)
        rows.append(m)
    if not rows:
        # fall back to the client log
        t = (d / "client.log").read_text()
        rows = [dict(re.findall(r"(\w+)=([\d.]+)", line)) for line in t.splitlines() if "repeat" in line and "rtfx" in line]
    use = rows[1:] if len(rows) > 1 else rows
    rtfx = [float(r["rtfx"]) for r in use if r.get("rtfx")]
    wer = [float(r["wer"]) for r in use if r.get("wer") is not None]
    return {"rtfx": statistics.mean(rtfx) if rtfx else None, "wer": statistics.mean(wer) if wer else None,
            "n": len(rtfx)}


for run in sys.argv[2:]:
    rd = root / run
    print(f"## {run}")
    for arm in sorted(p.name for p in rd.iterdir() if p.is_dir()):
        perf = rd / arm / "perf"
        if not perf.exists():
            continue
        if (perf / "asr_c1").exists():
            parts = []
            for c in (1, 16, 64):
                a = asr(perf / f"asr_c{c}")
                if a:
                    parts.append(f"c{c} RTFx {a['rtfx']:.0f} WER {100*a['wer']:.2f}%")
            print(f"  {arm:8s} " + " | ".join(parts))
            continue
        parts = []
        for w in ("i2t_c32", "t2t_c32", "i2t_c1"):
            m = runner_metrics(perf / w / "client.log")
            if m and m["tok_s"]:
                parts.append(f"{w} {m['tok_s']:.0f} tok/s ttft50 {m['ttft_p50_ms']:.0f}ms p95 {m['ttft_p95_ms']:.0f}")
        for w in ("big_c8", "huge_c4", "xl_c4", "phone_c4"):
            q = qcheck(perf / w)
            if q:
                parts.append(f"{w} {q['req_per_s']:.1f} req/s ttft50 {1e3*q['ttft_p50']:.0f}ms")
        q = qcheck(perf / "check")
        if q:
            parts.append("c1 ttft50 by kind " + ",".join(f"{k}:{1e3*v:.0f}" for k, v in q["ttft_p50_by_kind"].items()))
        for tag in ("t2t", "i2t"):
            m = runner_metrics(perf / "mixed" / f"{tag}.log")
            if m and m["tok_s"]:
                parts.append(f"mixed.{tag} {m['tok_s']:.0f} tok/s ttft50 {m['ttft_p50_ms']:.0f}ms")
        print(f"  {arm:8s} " + " | ".join(parts))
