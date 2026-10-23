"""One-off, on node8: once Qwen3-Omni's entry completes, drop its A2T (speech recognition) results so the
runner re-does them with every engine getting the same explicit system message (client.py fix of
2026-10-22; before it, M* applied Qwen3-Omni's voice persona and answered spoken questions instead of
transcribing). Also drops the M* main t2s c=1 r1 cell the old watchdog cut short. Completed cells of other
workloads stay; the runner returns to the entry at its next model boundary and fills in what's missing.

  setsid nohup python3 purge_omni_a2t.py > /workspace/results/bench/purge_omni_a2t.log 2>&1 &
"""
import json
import os
import shutil
import time
from pathlib import Path

OUT = Path("/workspace/results/bench/qwen3_omni")
DONE = OUT / "stages" / "model_complete.done"

while not DONE.exists():
    time.sleep(30)
time.sleep(5)
print("qwen3_omni complete:", time.strftime("%H:%M:%S"), flush=True)

res = OUT / "results.jsonl"
rows = [json.loads(line) for line in res.read_text().splitlines() if line.strip()]


def drop(r):
    if r.get("workload") == "omni_a2t":
        return True
    return (r.get("workload") == "omni_t2s" and r.get("arm") == "mstar-main" and r.get("conc") == 1
            and r.get("round") == 1 and r.get("failed") != 0)


keep = [r for r in rows if not drop(r)]
shutil.copy(res, OUT / "results.pre_a2t_purge.jsonl")
res.with_suffix(".tmp").write_text("".join(json.dumps(r) + "\n" for r in keep))
os.replace(res.with_suffix(".tmp"), res)
print(f"results: dropped {len(rows) - len(keep)} rows", flush=True)

old = OUT / "pre_a2t_purge"
old.mkdir(exist_ok=True)
for p in [OUT / "gate_a2t", *OUT.glob("wer_a2t_*.json"), OUT / "stages" / "gate_verdicts.json",
          OUT / "stages" / "gate_annotated.done", DONE]:
    if p.exists():
        shutil.move(str(p), str(old / p.name))
        print("moved aside:", p.name, flush=True)
print("done; the runner re-does the A2T gate and cells at its next model boundary", flush=True)
