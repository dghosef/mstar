"""Kernel breakdown of an nsys sqlite export by worker NVTX node/walk range.

  analyze_nsys.py c64.sqlite [--top 25]

Kernels are attributed to the innermost `worker[..].node[..].graph_walk[..]` NVTX range that encloses the
CUDA runtime call (kernel launch or graph launch) that issued them, on the launching thread.
Prints: capture span, GPU busy (union of kernel intervals), per node/walk: steps, GPU ms per step, kernels
per step, share of kernel time; and the top kernels per node/walk.
"""
import argparse
import bisect
import re
import sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--top", type=int, default=20)
a = ap.parse_args()
con = sqlite3.connect(a.db)
q = con.execute

def table_exists(t):
    return q("select count(*) from sqlite_master where type='table' and name=?", (t,)).fetchone()[0] > 0

strings = dict(q("select id, value from StringIds"))
kern = q("select start, end, correlationId, demangledName, shortName, deviceId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
GRAPH_ID = -12345
strings[GRAPH_ID] = "<cuda graph replay>"
if table_exists("CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
    kern += [(s, e, c, GRAPH_ID, GRAPH_ID, d) for s, e, c, d in
             q("select start, end, correlationId, deviceId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
if table_exists("CUPTI_ACTIVITY_KIND_MEMCPY"):
    strings[GRAPH_ID - 1] = "<memcpy>"
    kern += [(s, e, c, GRAPH_ID - 1, GRAPH_ID - 1, d) for s, e, c, d in
             q("select start, end, correlationId, deviceId from CUPTI_ACTIVITY_KIND_MEMCPY")]
if not kern:
    raise SystemExit("no kernels")
t0 = min(k[0] for k in kern); t1 = max(k[1] for k in kern)
# union of intervals
iv = sorted((k[0], k[1]) for k in kern)
busy = 0; cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce:
        busy += ce - cs; cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
span = t1 - t0
print(f"capture kernel span {span/1e6:.1f} ms, GPU busy {busy/1e6:.1f} ms ({100*busy/span:.1f}%), kernels {len(kern)}")

rt = {}
for corr, start, tid in q("select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
    rt[corr] = (start, tid)

# NVTX push/pop ranges (eventType 59 = PushPop, 60 = StartEnd)
ranges = defaultdict(list)  # tid -> list of (start, end, name)
pat = re.compile(r"node\[(.*?)\]\.graph_walk\[(.*?)\]")
for start, end, text, textId, tid in q("select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"):
    name = text if text is not None else strings.get(textId, "")
    m = pat.search(name or "")
    if m:
        ranges[tid].append((start, end, f"{m.group(1)}/{m.group(2)}"))
idx = {}
for tid, rs in ranges.items():
    rs.sort()
    idx[tid] = ([r[0] for r in rs], rs)

def label(corr):
    r = rt.get(corr)
    if r is None:
        return "?"
    s, tid = r
    if tid not in idx:
        return "unlabeled"
    starts, rs = idx[tid]
    i = bisect.bisect_right(starts, s) - 1
    best = None
    while i >= 0 and i >= len(rs) - 10**9:
        st, en, nm = rs[i]
        if st <= s <= en:
            best = nm
            break
        if s - st > 5e9:
            break
        i -= 1
    return best or "unlabeled"

per = defaultdict(lambda: [0, 0])          # label -> [ns, count]
perk = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for s, e, corr, dname, sname, dev in kern:
    lb = label(corr)
    per[lb][0] += e - s; per[lb][1] += 1
    nm = strings.get(sname) or strings.get(dname) or "?"
    perk[lb][nm][0] += e - s; perk[lb][nm][1] += 1

steps = defaultdict(int)
for tid, (_, rs) in idx.items():
    for st, en, nm in rs:
        if t0 <= st <= t1:
            steps[nm] += 1
tot = sum(v[0] for v in per.values())
print(f"{'node/walk':40s} {'steps':>6s} {'GPU ms':>9s} {'share':>6s} {'ms/step':>8s} {'kern/step':>9s}")
for lb, (ns, cnt) in sorted(per.items(), key=lambda x: -x[1][0]):
    st = steps.get(lb, 0)
    print(f"{lb:40s} {st:6d} {ns/1e6:9.1f} {100*ns/tot:5.1f}% {ns/1e6/max(st,1):8.3f} {cnt/max(st,1):9.1f}")
for lb, ks in sorted(perk.items(), key=lambda x: -per[x[0]][0]):
    print(f"\n== top kernels: {lb}")
    lt = per[lb][0]
    for nm, (ns, cnt) in sorted(ks.items(), key=lambda x: -x[1][0])[: a.top]:
        print(f"  {100*ns/lt:5.1f}% {ns/1e6:8.2f} ms  n={cnt:6d}  {ns/cnt/1e3:7.2f} us  {nm[:110]}")

# GPU idle gaps between consecutive kernels, attributed to the (previous label -> next label) transition
ks = sorted(((s, e, label(c)) for s, e, c, *_ in kern), key=lambda x: x[0])
gaps = defaultdict(lambda: [0, 0])
ce = ks[0][1]; cl = ks[0][2]
for s, e, lb in ks[1:]:
    if s > ce:
        g = s - ce
        if g > 20_000:  # >20 us
            key = f"{cl} -> {lb}"
            gaps[key][0] += g; gaps[key][1] += 1
    if e > ce:
        ce, cl = e, lb
print(f"\n== idle gaps >20us by transition (total idle {(span-busy)/1e6:.1f} ms)")
for k, (ns, n) in sorted(gaps.items(), key=lambda x: -x[1][0])[:12]:
    print(f"  {ns/1e6:8.1f} ms  n={n:5d}  mean {ns/n/1e3:7.1f} us  {k}")
