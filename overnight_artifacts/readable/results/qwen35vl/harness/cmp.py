"""Compare qcheck texts between two result dirs: per id, fraction of rows identical; and first-diff position."""
import json, sys
def load(d):
    rows = json.load(open(f"{d}/results.json"))["rows"]
    out = {}
    for r in rows:
        if "text" in r:
            out.setdefault(r["id"], []).append(r["text"])
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
same = tot = 0
diffs = []
for k in sorted(a):
    for x in a[k]:
        for y in b.get(k, []):
            tot += 1
            if x == y:
                same += 1
            else:
                p = next((i for i, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
                diffs.append((k, p))
ids_same = sum(1 for k in a if k in b and a[k][0] == b[k][0])
print(f"pairs identical {same}/{tot}; ids identical (first row) {ids_same}/{len(a)}; diffs (id, first diff char) {sorted(set(diffs))[:12]}")
