"""nsys window of one cell: server under `nsys launch` (worker NVTX on), collection only around the measured cell.

  nsysrun.py --arm name=/tree[:env] --model qwen3_5_0.8b --gpus 0 --cell random_1024_256:64 --n 512 --out DIR
Writes DIR/<name>.nsys-rep, DIR/<name>.sqlite, DIR/<name>_kern.csv (cuda_gpu_kern_sum), DIR/<name>_top.txt and
DIR/<name>_analyze.txt (perf-engineering's analyze_nsys.py: GPU busy, per node/walk ms and kernels per step).
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab  # noqa: E402

ANALYZE = Path(__file__).resolve().parent / "analyze_nsys.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="qwen3_5_0.8b")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--port", type=int, default=59100)
    ap.add_argument("--cell", default="random_1024_256:64")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--graph-trace", default="graph", choices=["graph", "node"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    name, tree, env = ab.parse_arms(a.arm)[0]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sess = f"p4q_{a.port}"
    env["OVN_WRAP"] = (f"nsys launch --session-new={sess} --trace=cuda,nvtx --cuda-graph-trace={a.graph_trace} "
                       "--trace-fork-before-exec=true")
    # worker NVTX ranges where the tree's CLI has the flag; kernels are captured either way
    if tree not in ("vllm", "sglang") and "--enable-nvtx" in (Path(tree) / "mstar/cli/main.py").read_text():
        env["OVN_SERVE_EXTRA"] = (env.get("OVN_SERVE_EXTRA", "") + " --enable-nvtx").strip()
    wl, c = a.cell.split(":")
    s = ab.Server(name, tree, env, a.model, a.gpus, a.port, out / f"{name}_server.log")
    rep = out / name
    try:
        ab.run_cell(a.model, a.port, wl, int(c), 2 * int(c), 2, out / f"{name}_warm")
        subprocess.run(["nsys", "start", f"--session={sess}", "-o", str(rep), "--force-overwrite=true"], check=True)
        row = ab.run_cell(a.model, a.port, wl, int(c), a.n, 0, out / f"{name}_cell")
        subprocess.run(["nsys", "stop", f"--session={sess}"], check=True)
        ab.log(f"nsys cell {name}: {row}")
    finally:
        s.stop()
    pin = ["numactl", f"--physcpubind={ab.CORES}"]
    subprocess.run(pin + ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o",
                          str(rep) + ".sqlite", str(rep) + ".nsys-rep"], check=False, capture_output=True)
    r = subprocess.run(pin + ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "-o", "-",
                              str(rep) + ".sqlite"], check=False, capture_output=True, text=True)
    (out / f"{name}_kern.csv").write_text(r.stdout)
    rows = list(csv.DictReader(io.StringIO(r.stdout[r.stdout.find("Time (%)"):]))) if "Time (%)" in r.stdout else []
    lines = [f"{float(x['Time (%)']):6.2f}%  {float(x['Total Time (ns)']) / 1e6:9.1f} ms  "
             f"n={x['Instances']:>7s}  {x['Name'][:150]}" for x in rows[:45]]
    (out / f"{name}_top.txt").write_text("\n".join(lines) + "\n" + r.stderr[-2000:])
    if ANALYZE.exists():
        r2 = subprocess.run(pin + [ab.PY, str(ANALYZE), str(rep) + ".sqlite", "--top", "25"],
                            check=False, capture_output=True, text=True)
        (out / f"{name}_analyze.txt").write_text(r2.stdout + r2.stderr[-3000:])
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
