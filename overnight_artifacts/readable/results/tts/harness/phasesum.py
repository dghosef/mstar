"""Print the largest steady-state phase segments per worker from a server log.

    python phasesum.py server.log [--top 2] [--min-records 3] [--all-phases]
"""
import argparse
import sys

sys.path.insert(0, "/root/work/mstar")
from benchmark.worker_phases.phases import parse_log, render, segment  # noqa: E402

KEY = [
    "iter_total", "gpu_exec", "worker.gpu_thread.exec", "worker.gpu_thread.prepare_inputs",
    "await_gpu", "worker.postprocess.event_sync", "worker.postprocess_batch", "worker.postprocess.check_stop",
    "worker.postprocess.register_outputs", "worker.schedule", "engine.forward", "engine.preprocess",
    "engine.postprocess", "engine.plan.promoted", "engine.plan.fresh", "submit_spec", "worker.gpu_thread.await_plan",
    "worker.plan_thread.pre_plan", "engine.collect_outputs", "worker.postprocess.route_outputs",
]

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--top", type=int, default=2)
ap.add_argument("--min-records", type=int, default=3)
ap.add_argument("--all-phases", action="store_true")
ap.add_argument("--tol", type=float, default=0.25)
a = ap.parse_args()
segs = segment(parse_log(open(a.log).read()), skip_warmup=2, bs_tolerance=a.tol, min_records=a.min_records)
by = {}
for s in segs:
    by.setdefault(s.worker, []).append(s)
pick = []
for w, ss in sorted(by.items()):
    pick += sorted(ss, key=lambda s: -len(s.records))[: a.top]
print(render(pick, None if a.all_phases else KEY))
