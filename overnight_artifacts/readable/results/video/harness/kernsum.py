import csv, glob, sys
f = glob.glob(sys.argv[1] + "/kern*.csv")[0]
rows = list(csv.DictReader(open(f)))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
print("total kernel ms", round(tot / 1e6, 1))
for r in rows[: int(sys.argv[2]) if len(sys.argv) > 2 else 25]:
    print("%5.1f%% %9.1fms n=%6s %s" % (float(r["Time (%)"]), float(r["Total Time (ns)"]) / 1e6, r["Instances"], r["Name"][:120]))
