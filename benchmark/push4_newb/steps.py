"""Summarize OVN_STEPLOG step logs: what step kinds ran, how many tokens, and how long until the next step.

  python steps.py <dir> [--node LLM] [--t0 SEC] [--t1 SEC]   (window relative to the first record)
"""
import argparse
import collections
import glob
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--node", default="LLM")
    ap.add_argument("--skip-frac", type=float, default=0.3, help="drop the first fraction of records (warmup)")
    a = ap.parse_args()
    for path in sorted(glob.glob(f"{a.dir}/steps_*.tsv")):
        rows = []
        for line in open(path):
            t, node, walk, bs, ntok, b = line.rstrip("\n").split("\t")
            if node != a.node:
                continue
            rows.append((float(t), walk, int(bs), None if ntok == "None" else int(ntok), b))
        if len(rows) < 10:
            continue
        rows = rows[int(len(rows) * a.skip_frac):]
        span = rows[-1][0] - rows[0][0]
        kinds = collections.defaultdict(list)
        for (t, walk, bs, ntok, b), nxt in zip(rows, rows[1:]):
            # packed steps are leased with num_tokens; batched (decode) with None
            k = (walk, "eager" if b == "eager" else ("graph" if ntok is not None else "batched"))
            kinds[k].append((nxt[0] - t, bs, ntok or 0, b))
        print(f"== {path}: {len(rows)} leases over {span:.1f}s")
        print(f"{'walk':12s} {'kind':8s} {'n':>6s} {'share_t':>7s} {'mean_ms':>8s} {'p50_ms':>7s} {'bs':>6s} {'tok':>7s}  buckets")
        for k, v in sorted(kinds.items(), key=lambda kv: -sum(x[0] for x in kv[1])):
            dt = [x[0] * 1000 for x in v]
            bks = collections.Counter(x[3] for x in v).most_common(4)
            real = sum(x[2] for x in v)
            padded = sum(int(x[3].split("x")[1]) for x in v if "x" in x[3])
            pad = f"  real/bucket tokens {real}/{padded} = {real / padded:.2f}" if padded else ""
            print(f"{k[0]:12s} {k[1]:8s} {len(v):6d} {sum(dt) / 1000 / span:7.2f} {statistics.mean(dt):8.2f} "
                  f"{statistics.median(dt):7.2f} {statistics.mean(x[1] for x in v):6.1f} "
                  f"{statistics.mean(x[2] for x in v):7.0f}  {bks}{pad}")


if __name__ == "__main__":
    main()
