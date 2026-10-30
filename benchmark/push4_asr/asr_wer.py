"""WER of nemo_bench.py's hypotheses with benchmark.asr_bench's normalizer and scorer (same as the served runs).

  PYTHONPATH=<tree> python asr_wer.py results/asr/<tag>/nemo_<model>/nemo.json
"""
import json
import sys
from pathlib import Path

from benchmark.asr_bench import make_normalizer, word_error_rate

res = json.loads(Path(sys.argv[1]).read_text())
d = Path.home() / "push4/asr/data/test_clean_200"
refs = dict(line.split("\t", 1) for line in (d / "references.tsv").read_text().splitlines() if line.strip())
norm = make_normalizer("whisper")
for p in res["points"]:
    uids = list(refs)
    wer = word_error_rate([refs[u] for u in uids], [p["hyps"].get(u, "") for u in uids], norm)
    print(f"{res['model']} batch {p['batch']}: RTFx {p['rtfx']} WER {wer * 100:.2f}%")
