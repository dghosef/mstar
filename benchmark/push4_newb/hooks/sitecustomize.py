"""Opt-in diagnostics for overnight-2 experiments (loaded via PYTHONPATH=<this dir>:<tree>).

OVN_STEPLOG=<dir>: every AcceleratorGraphRunner.lease_slot call (walk, bs, num_tokens, chosen bucket or
eager) with a monotonic timestamp, written as <dir>/steps_<pid>.tsv in batches.
MSTAR_TPROF=1: torch-profiler CUDA kernel table over MSTAR_TPROF_STEPS graph replays after MSTAR_TPROF_SKIP.
"""
import os
import sys

_steplog = os.environ.get("OVN_STEPLOG")
if _steplog:
    import atexit
    import threading
    import time

    _rows: list = []
    _lk = threading.Lock()
    _path = os.path.join(_steplog, f"steps_{os.getpid()}.tsv")

    def _flush():
        with _lk:
            rows, _rows[:] = list(_rows), []
        if rows:
            with open(_path, "a") as f:
                f.write("".join(rows))

    atexit.register(_flush)

    def _install():
        try:
            from mstar.engine import accelerator_graph_runner as agr
        except Exception as e:  # noqa: BLE001
            print(f"OVN_STEPLOG hook failed: {e!r}", file=sys.stderr)
            return
        orig = agr.AcceleratorGraphRunner.lease_slot

        def lease_slot(self, graph_walk, bs, slot, num_tokens=None, cg_key_info=None):
            r = orig(self, graph_walk, bs, slot, num_tokens=num_tokens, cg_key_info=cg_key_info)
            b = "eager" if r is None or r.bucket is None else f"{r.bucket.bs}x{r.bucket.num_tokens}"
            with _lk:
                _rows.append(f"{time.monotonic():.6f}\t{self._submodule_name}\t{graph_walk}\t{bs}\t{num_tokens}\t{b}\n")
                n = len(_rows)
            if n >= 2000:
                _flush()
            return r

        agr.AcceleratorGraphRunner.lease_slot = lease_slot

    _install()

if os.environ.get("MSTAR_TPROF") == "1":
    import threading

    import torch

    _skip = int(os.environ.get("MSTAR_TPROF_SKIP", "500"))
    _steps = int(os.environ.get("MSTAR_TPROF_STEPS", "200"))
    _out = os.environ.get("MSTAR_TPROF_OUT", "/tmp")
    _state = {"n": 0, "prof": None, "done": False}
    _lock = threading.Lock()
    _orig = torch.cuda.CUDAGraph.replay

    def _replay(self, *a, **k):
        with _lock:
            _state["n"] += 1
            n = _state["n"]
            if not _state["done"] and n == _skip:
                p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                p.__enter__()
                _state["prof"] = p
        r = _orig(self, *a, **k)
        with _lock:
            if _state["prof"] is not None and n == _skip + _steps:
                torch.cuda.synchronize()
                p = _state["prof"]
                _state["prof"], _state["done"] = None, True
                p.__exit__(None, None, None)
                table = p.key_averages().table(sort_by="cuda_time_total", row_limit=60, max_name_column_width=90)
                with open(os.path.join(_out, f"tprof_{os.getpid()}.txt"), "w") as f:
                    f.write(f"replays profiled: {_steps}\n{table}\n")
        return r

    torch.cuda.CUDAGraph.replay = _replay
