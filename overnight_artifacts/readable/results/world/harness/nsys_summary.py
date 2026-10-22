"""Summarize an nsys-rep's kernels: busy %, top kernels by time."""
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def summarize(rep: Path, header: str, units: int) -> str:
    out = rep.parent
    subprocess.run(["nsys", "export", "--type=sqlite", "-o", str(out / "trace.sqlite"), "-f", "true",
                    str(rep)], check=True, capture_output=True)
    db = sqlite3.connect(out / "trace.sqlite")
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = db.execute("SELECT start, end, shortName, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    by = defaultdict(lambda: [0, 0])
    ivs = sorted((s, e) for s, e, _, _ in rows)
    busy, cur_s, cur_e = 0, None, None
    for s, e in ivs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (ivs[-1][1] - ivs[0][0]) if ivs else 1
    for s, e, n, _ in rows:
        by[names.get(n, str(n))][0] += e - s
        by[names.get(n, str(n))][1] += 1
    total = sum(v[0] for v in by.values())
    lines = [f"{header} kernel_span={span/1e9:.3f}s busy={busy/1e9:.3f}s ({100*busy/span:.1f}%) "
             f"kernels={len(rows)} kernel_sum={total/1e9:.3f}s per_unit_ms={total/1e6/max(units,1):.2f}"]
    for n, (t, c) in sorted(by.items(), key=lambda kv: -kv[1][0])[:50]:
        lines.append(f"{100*t/total:6.2f}% {t/1e6:9.2f}ms n={c:6d} avg={t/c/1e3:8.1f}us {n[:120]}")
    (out / "kernels.txt").write_text("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])))
