"""Correctness gate: PASS/FAIL per check, exit 1 on any FAIL. Benchmarks run only on PASS.

The thresholds are the defaults below unless overridden, and they are fixed (in the
harness's docstring and here) BEFORE the run, not tuned to the result.

Checks, each enabled by the input it needs:

  --pytest-log LOG   the real-checkpoint pytest run: a summary line exists, nothing
                     failed or errored, and nothing skipped unless --allow-skips
                     (a skipped decode/real-weights test is a check that didn't run).
  --parity DIR       parity.py's report.json. The gated run (--tag) must have
                       mean argmax agreement >= --min-agreement,
                       and >= (worst peer engine) - --peer-slack,
                       agreement == --min-confident (1.0) at confident positions (oracle top-2
                         margin > 1; Qwen3-TTS #302's gate, robust to bf16 near-ties),
                       every first disagreement a near-tie: gap <= max(--max-gap, worst peer gap)
                         and the served token within the oracle's top --topk (vLLM's rule),
                       prompt_tokens_delta == 0 on every prompt that reports it (tokenization/template
                         drift; mstar reports no usage, so this prints UNCHECKED for it),
                       no empty completion.
                     --control (same code, second server process) is printed as the noise floor;
                     if the gated run is far below the control, that is reported too.
  --json-check NAME=PATH:KEY<=VALUE / KEY>=VALUE   a scalar in some other JSON result (e.g.
                     component parity: 'vision=out/vision.json:fp32_cosine>=0.9999'). KEY may be dotted.

  python gate.py --pytest-log out/pytest.log --parity out/parity --tag mstar_a --control mstar_b \\
      --peers vllm,sglang --json-check 'vision=out/vision_parity.json:fp32_cosine>=0.9999'
"""
import argparse
import json
import re
import sys
from pathlib import Path


def pytest_check(path: Path, allow_skips: bool) -> tuple[bool, str]:
    text = path.read_text() if path.exists() else ""
    summary = (re.findall(r"^=*\s*(\d+ (?:passed|failed|error|skipped)[^=\n]*? in [\d.]+s)", text, re.M) or [""])[-1]
    ok = (bool(summary) and "failed" not in summary and "error" not in summary
          and (allow_skips or "skipped" not in summary))
    return ok, summary or f"no pytest summary in {path}"


def parity_stats(rows: dict) -> dict:
    vals = list(rows.values())
    gaps = [v["logit_gap_at_first"] for v in vals if v.get("logit_gap_at_first") is not None]
    ranks = [v["rank_at_first"] for v in vals if v.get("rank_at_first") is not None]
    return {"n": len(vals), "mean": sum(v["argmax_agreement"] for v in vals) / len(vals),
            "max_gap": max(gaps, default=0.0), "max_rank": max(ranks, default=0),
            "deltas": [v.get("prompt_tokens_delta") for v in vals],
            "confident": [v["confident_agreement"] for v in vals if v.get("confident_agreement") is not None],
            "empty": sum(bool(v.get("empty")) for v in vals)}


def parity_checks(a) -> list[tuple[str, bool, str]]:
    path = Path(a.parity) / "report.json"
    if not path.exists():
        return [("teacher-forced parity", False, f"no {path}")]
    served = json.loads(path.read_text())["served"]
    if a.tag not in served:
        return [("teacher-forced parity", False, f"no '{a.tag}' run in {path} (have {sorted(served)})")]
    me = parity_stats(served[a.tag])
    peers = {p: parity_stats(served[p]) for p in a.peers.split(",") if p and p in served}
    missing = [p for p in a.peers.split(",") if p and p not in served]
    floor = min([s["mean"] for s in peers.values()], default=1.0) - a.peer_slack
    gap_cap = max([a.max_gap] + [s["max_gap"] for s in peers.values()])
    yard = "; ".join(f"{t} {s['mean']:.4f} gap {s['max_gap']:.2f}" for t, s in peers.items())
    if a.control in served:
        c = parity_stats(served[a.control])
        yard = f"control {a.control} {c['mean']:.4f} gap {c['max_gap']:.2f}; " + yard
    checks = [
        ("argmax agreement", me["mean"] >= a.min_agreement and me["mean"] >= floor,
         f"{a.tag} {me['mean']:.4f} over {me['n']} prompts (need >= {a.min_agreement} and >= {floor:.4f}); {yard}"
         + (f"; MISSING peers {missing}" if missing else "")),
        ("first disagreements are near-ties", me["max_gap"] <= gap_cap and me["max_rank"] < a.topk,
         f"max gap {me['max_gap']:.2f} (cap {gap_cap:.2f}), max rank {me['max_rank']} (need < {a.topk})"),
        ("confident positions all agree", bool(me["confident"]) and min(me["confident"]) >= a.min_confident - 1e-6,
         f"min over prompts {min(me['confident']):.4f} (need >= {a.min_confident})" if me["confident"]
         else "no confident positions recorded (old report.json?)"),
        ("prompt tokens match the oracle", all(d == 0 for d in me["deltas"] if d is not None),
         f"served - oracle per prompt: {me['deltas']}" if any(d is not None for d in me["deltas"]) else
         "UNCHECKED: the server reports no usage.prompt_tokens; a prompt-ids test vs the HF processor must cover this"),
        ("no empty completions", me["empty"] == 0, f"{me['empty']} empty"),
    ]
    return [(f"parity: {name}", ok, detail) for name, ok, detail in checks]


def json_check(spec: str) -> tuple[str, bool, str]:
    m = re.fullmatch(r"([^=]+)=([^:]+):([\w.\- ]+?)(<=|>=)([-+.\deE]+)", spec)
    if not m:
        return (spec, False, "bad --json-check spec (NAME=PATH:KEY<=VALUE)")
    name, path, key, op, bound = m.groups()
    try:
        val = json.loads(Path(path).expanduser().read_text())
        for part in key.split("."):
            val = val[part]
        val = float(val)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return (name, False, f"cannot read {key} from {path}: {e!r}")
    ok = val <= float(bound) if op == "<=" else val >= float(bound)
    return (name, ok, f"{key} = {val:.6g} (need {op} {bound})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytest-log", type=Path)
    ap.add_argument("--allow-skips", action="store_true")
    ap.add_argument("--parity")
    ap.add_argument("--tag", default="mstar_a")
    ap.add_argument("--control", default="mstar_b")
    ap.add_argument("--peers", default="vllm,sglang")
    ap.add_argument("--min-agreement", type=float, default=0.95)
    ap.add_argument("--peer-slack", type=float, default=0.02)
    ap.add_argument("--max-gap", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min-confident", type=float, default=1.0,
                    help="agreement required where the oracle's top-2 margin > 1 (#302)")
    ap.add_argument("--json-check", action="append", default=[])
    a = ap.parse_args()

    checks: list[tuple[str, bool, str]] = []
    if a.pytest_log:
        ok, detail = pytest_check(a.pytest_log, a.allow_skips)
        checks.append(("pytest with the real checkpoint", ok, detail))
    if a.parity:
        checks += parity_checks(a)
    checks += [json_check(s) for s in a.json_check]
    if not checks:
        sys.exit("nothing to gate: pass --pytest-log, --parity and/or --json-check")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    passed = all(ok for _, ok, _ in checks)
    print("GATE", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
