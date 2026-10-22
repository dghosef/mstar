#!/usr/bin/env python3
"""Run the CI CPU Core suite (+ TP lockstep sim) against one tree on the box; write the failing-test set.

    python cputests.py --tree /root/work/arms/cand --out /workspace/results/integrator/tests/cand [-n 32]
"""
import argparse
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

VENV = os.environ.get("REGRESS_VENV", "/root/work/mstar/.venv")
CORE = ("--ignore=test/modular/test_tp_async_lockstep_sim.py --ignore=test/modular/test_piecewise_config_signature.py test/modular test/distributed test/chatterbox "
        "test/kokoro test/omnivoice test/sampling_test mstar/model/*/tests benchmark/worker_phases")


def run(tree, out, n, label, paths):
    env = dict(os.environ, PYTHONPATH=tree, HF_HUB_OFFLINE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0",
               PATH=f"{VENV}/bin:" + os.environ["PATH"], VIRTUAL_ENV=VENV)
    xml = out / f"{label}.xml"
    xd = f"-p xdist -n {n} --dist worksteal " if n > 1 else ""
    cmd = (f"{VENV}/bin/python -m pytest -q -ra --strict-markers {xd}"
           f"--basetemp=/tmp/pt_{label}_{os.getpid()} --junitxml={xml} {paths}")
    t0 = time.time()
    with open(out / f"{label}.log", "w") as log:
        rc = subprocess.call(cmd, shell=True, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT)
    fails, total = [], 0
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            total += 1
            if tc.find("failure") is not None or tc.find("error") is not None:
                fails.append(f"{tc.get('classname')}::{tc.get('name')}")
    return {"rc": rc, "secs": round(time.time() - t0), "total": total, "failed": sorted(fails)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"core": run(a.tree, out, a.n, "core", CORE)}
    # its parametrize ids come from __subclasses__(), whose order differs between xdist workers
    res["piecewise_sig"] = run(a.tree, out, 1, "piecewise_sig", "test/modular/test_piecewise_config_signature.py")
    if not a.skip_sim:
        res["tp_sim"] = run(a.tree, out, 3, "tp_sim", "test/modular/test_tp_async_lockstep_sim.py")
    (out / "tests.json").write_text(json.dumps(res, indent=1))
    (out / "DONE").write_text("done")
    print(json.dumps({k: {**v, "failed": v["failed"][:50]} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
