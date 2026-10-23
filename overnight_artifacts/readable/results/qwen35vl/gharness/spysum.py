"""Summarize py-spy raw (collapsed) output: per thread, samples and top self/inclusive frames."""
import sys, collections
for path in sys.argv[1:]:
    per_thread = collections.Counter(); incl = collections.defaultdict(collections.Counter); leaf = collections.defaultdict(collections.Counter)
    for line in open(path):
        line = line.rstrip()
        if not line: continue
        stack, n = line.rsplit(" ", 1); n = int(n)
        frames = stack.split(";")
        th = frames[0]
        per_thread[th] += n
        seen = set()
        for f in frames[1:]:
            fn = f.split(" (")[0]
            if fn not in seen:
                incl[th][fn] += n; seen.add(fn)
        leaf[th][frames[-1][:110]] += n
    print("=====", path)
    for th, tot in per_thread.most_common(6):
        print(f"-- {th} samples={tot}")
        for f, n in incl[th].most_common(14):
            print(f"   {100*n/tot:5.1f}% incl {f}")
        for f, n in leaf[th].most_common(5):
            print(f"   {100*n/tot:5.1f}% leaf {f}")
