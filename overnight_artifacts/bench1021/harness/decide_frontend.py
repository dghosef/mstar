"""Pick M*'s HTTP frontend for a model from the frontend tune, by a rule fixed beforehand:
Rust if its correctness gate passed and its geomean tok/s beats Python's by more than 3%
(the tuning noise), else Python. Writes tuned.json "mstar_frontend" and prints the arm
names to use: "<main arm> <pr arm>".

  python decide_frontend.py /workspace/results/bench/q35_0.8b
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
t = json.loads((d / "tuned.json").read_text())
score = t.get("mstar_frontend_geomean", {})
gate = (d / "gate.txt").read_text() if (d / "gate.txt").exists() else ""
blk = gate.split("#### gate mstar_pr_rf_a")[1].split("####")[0] if "#### gate mstar_pr_rf_a" in gate else ""
rust_ok = "GATE PASS" in blk
py, rs = score.get("mstar-pr"), score.get("mstar-pr-rf")
choice = "rust" if (rust_ok and py and rs and rs > py * 1.03) else "python"
t["mstar_frontend"] = choice
t["mstar_frontend_reason"] = (f"python {py and round(py)} vs rust {rs and round(rs)} geomean tok/s; "
                              f"rust gate {'PASS' if rust_ok else 'FAIL/missing'}; rule: rust if gate passes and > +3%")
(d / "tuned.json").write_text(json.dumps(t, indent=1))
sfx = "-rf" if choice == "rust" else ""
print(f"mstar-main{sfx} mstar-pr{sfx}")
