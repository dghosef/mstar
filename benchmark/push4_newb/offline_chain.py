"""push4-newb's unattended queue on push4c GPUs 0-3 (the user is offline; nothing on the Mac is needed).

Jobs run in order, each under a hard timeout with its own log; a failed or timed-out job is logged and the next one
starts. Jobs marked `all_gpus` first wait until GPUs 1-3 have no compute processes (they are lent out until ~12:40Z).

  nohup setsid python offline_chain.py [--jobs=a,b] [PID ...] >> ~/push4/results/newb/offline_chain.log 2>&1 &
  (PIDs: jobs already on GPU 0 to wait for first)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/root/push4/newb/venv/bin/python"
R = Path.home() / "push4" / "results" / "newb"
LADDER = "benchmark/push4_newb/cfg/q27b_tp4_async_knobs_ladder.yaml"
TREE = "/root/push4/newb/mstar"


def evens(a: int, b: int) -> str:
    return ",".join(str(c) for c in range(a, b + 1, 2))


GEN = {"PUSH4_MSTAR_CORES": "0,2", "PUSH4_CORES": evens(4, 52), "HF_HOME": "/root/hf"}
TP4 = {"PUSH4_CORES": evens(0, 206), "HF_HOME": "/root/hf",
       "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"}
NSYS = [PY, "nsysrun.py", "--model", "qwen3_5_27b_tp4", "--gpus", "0,1,2,3", "--port", "59100",
        "--cell", "random_1024_256:128", "--n", "256", "--out", str(R / "q27b_nsys")]
BENCH27 = [PY, "run_27b.py", "--phases", "bench", "--arms", "ladder,vllm,sglang"]
# prefill only (1 output token): mstar's prefill is 6-11% slower per token than vLLM's (q27b_prefill)
NSYS_PF = [PY, "nsysrun.py", "--model", "qwen3_5_27b_tp4", "--gpus", "0,1,2,3", "--port", "59100",
           "--cell", "random_1024_1:128", "--n", "512", "--out", str(R / "q27b_nsys_prefill")]

HEAD = "/root/push4/newb/head"
# head's 27B ShareGPT gap is decode (TPOT 25.8 vs vLLM 21.0 ms at c128): phase timing, then an nsys pair, same cell
NSYS_SG = [PY, "nsysrun.py", "--model", "qwen3_5_27b_tp4", "--gpus", "0,1,2,3", "--port", "59100",
           "--cell", "sharegpt:128", "--n", "512", "--out", str(R / "q27b_sg_nsys")]

KV = R / "kvpc" / "cfg"
KVPC_27B_ARMS = ",".join(f"{n}={HEAD}:OVN_CONFIG={c}" for n, c in (
    ("none", KV / "27b_none.yaml"), ("kvpc", KV / "27b_kvpc.yaml"), ("gc", KV / "27b_gc.yaml"),
    ("both", f"{HEAD}/configs/qwen3_5_27b_tp4.yaml")))

SHARED = "/root/push4/newb/shared"  # perf/push4-shared 3daafc851 (select_last_hidden fix)

# (name, command, timeout s, env, needs GPUs 1-3)
JOBS = [
    # q35text's prefill buckets 1152-1792 on 27B TP4 (a ~1040-token prompt padded to 2048): c1 TTFT + no regression
    ("buckets_27b", [PY, "ab.py", "bench", "--model", "qwen3_5_27b_tp4", "--rounds", "2",
                     "--cells", "random_1024_256:1,random_1024_256:128,sharegpt:1,sharegpt:128",
                     "--arms", f"shipped={SHARED},buckets={SHARED}:OVN_CONFIG={KV / '27b_buckets.yaml'}",
                     "--gpus", "0,1,2,3", "--port", "59100", "--out", str(R / "q27b_buckets")], 4 * 3600,
     {**TP4, "PUSH4_MSTAR_CORES": evens(0, 14), "PUSH4_CORES": evens(16, 206)}, True),
    # the knobs split on 27B TP4, reduced per main (9B showed no plan-cache effect): c1 + c128 random, 2 rounds
    ("kvpc_27b", [PY, "ab.py", "bench", "--model", "qwen3_5_27b_tp4", "--rounds", "2",
                  "--cells", "random_1024_256:1,random_1024_256:128", "--arms", KVPC_27B_ARMS,
                  "--gpus", "0,1,2,3", "--port", "59100",
                  "--out", str(R / "kvpc" / "q27b")], 4 * 3600,
     {**TP4, "PUSH4_MSTAR_CORES": evens(0, 14), "PUSH4_CORES": evens(16, 206)}, True),
    ("sg_profile_head", [PY, "ab.py", "profile", "--arms", f"head={HEAD}", "--cells", "sharegpt:128,sharegpt:64",
                         "--model", "qwen3_5_27b_tp4", "--gpus", "0,1,2,3", "--port", "59100",
                         "--out", str(R / "q27b_sg_profile")], 3600, TP4, True),
    ("sg_nsys_head", NSYS_SG[:2] + ["--arm", f"head={HEAD}"] + NSYS_SG[2:], 3600, TP4, True),
    ("sg_nsys_vllm", NSYS_SG[:2] + ["--arm", "vllm=vllm"] + NSYS_SG[2:], 3600, TP4, True),
    # pi0.5 is mstar's losing cell (JCT p50 ~120 ms vs openpi 35 ms in-process; mstar's worker is ~16 ms of it):
    # where the rest of mstar's request goes, then openpi's own websocket server so both arms pay for serving,
    # then the mstar server unpacked (is the host path CPU-starved on 2 CPUs?)
    # after 0da53b4ac (uint8 cameras, -24%): where the remaining API-side time goes
    ("pi05_timeline_u8", [PY, "run_pi05.py", "--gpu", "0", "--n", "50", "--timeline",
                          "--out", str(R / "pi05_timeline_u8")], 2400, GEN, False),
    ("pi05_hostprof_u8", [PY, "pi05_hostprof.py", "--gpu", "0", "--out", str(R / "pi05_hostprof_u8")],
     2400, GEN, False),
    ("pi05_timeline", [PY, "run_pi05.py", "--gpu", "0", "--n", "50", "--timeline", "--out", str(R / "pi05_timeline")],
     2400, GEN, False),
    ("pi05_ws", [PY, "run_pi05.py", "--gpu", "0", "--arms", "mstar,openpi_ws", "--rounds", "2", "--n", "50",
                 "--out", str(R / "pi05_ws")], 2 * 3600, GEN, False),
    ("pi05_wide", [PY, "run_pi05.py", "--gpu", "0", "--arms", "mstar", "--rounds", "1", "--n", "50",
                   "--out", str(R / "pi05_wide")], 2400, {**GEN, "PUSH4_MSTAR_CORES": evens(4, 52)}, False),
    # DROID third round through the compile-wide-then-repin launch (validates it against packed-from-start)
    ("droid_r3", [PY, "run_gen.py", "bench", "--model", "droid", "--gpu", "0", "--arms", "base,sglang_b1",
                  "--rounds", "1", "--out", str(R / "droid_r3")], 3600, GEN, False),
    # 27B TP4: where the TTFT gap to vLLM comes from (prefill kernel cost), then a workload not yet measured
    ("nsys_27b_ladder", NSYS[:2] + ["--arm", f"ladder={TREE}:OVN_CONFIG={LADDER}"] + NSYS[2:], 3600, TP4, True),
    ("nsys_27b_vllm", NSYS[:2] + ["--arm", "vllm=vllm"] + NSYS[2:], 3600, TP4, True),
    ("nsys_27b_ladder_retry", NSYS[:2] + ["--arm", f"ladder={TREE}:OVN_CONFIG={LADDER}"] + NSYS[2:], 3600, TP4, True),
    ("nsys_pf_ladder", NSYS_PF[:2] + ["--arm", f"ladder={TREE}:OVN_CONFIG={LADDER}"] + NSYS_PF[2:], 3600, TP4, True),
    ("nsys_pf_vllm", NSYS_PF[:2] + ["--arm", "vllm=vllm"] + NSYS_PF[2:], 3600, TP4, True),
    ("q27b_sharegpt", BENCH27 + ["--cells", "sharegpt:32,sharegpt:64,sharegpt:128", "--rounds", "2",
                                 "--out", str(R / "q27b_sharegpt")], 4 * 3600, TP4, True),
    ("q27b_random_r3", BENCH27 + ["--rounds", "1", "--out", str(R / "q27b_random_r3")], 2 * 3600, TP4, True),
]


def log(msg: str) -> None:
    print(f"[chain {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpus_1_3_free() -> bool:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id", "--format=csv,noheader"],
                       capture_output=True, text=True, check=False)
    busy = {x.strip() for x in r.stdout.splitlines() if x.strip()}
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                       capture_output=True, text=True, check=False)
    ours = {b.strip() for i, b in (x.split(",") for x in q.stdout.splitlines()) if i.strip() in ("1", "2", "3")}
    return not (busy & ours)


def wait_gpus_1_3(deadline_s: float = 4 * 3600) -> bool:
    t0, free_since = time.monotonic(), None
    while time.monotonic() - t0 < deadline_s:
        if gpus_1_3_free():
            free_since = free_since or time.monotonic()
            if time.monotonic() - free_since >= 60:  # a borrower's chain may restart a server between jobs
                return True
        else:
            free_since = None
        time.sleep(15)
    return False


def cleanup() -> None:
    """Kill leftover servers of this slice (own sessions, so a job's killpg misses them): processes under
    /root/push4/newb serving on one of push4-newb's ports. Other slices' processes never match."""
    me = os.getsid(0)
    r = subprocess.run(["ps", "-eo", "pid=,sid=,args="], capture_output=True, text=True, check=False)
    sids = set()
    for line in r.stdout.splitlines():
        pid, sid, args = line.split(None, 2) if len(line.split(None, 2)) == 3 else (None, None, "")
        if sid and int(sid) != me and "/root/push4/newb/" in args and any(
                f"--port {pt}" in args for pt in ("59100", "59150", "59200", "59300", "59400", "59500")):
            sids.add(int(sid))
    for sid in sids:
        log(f"cleanup: killing leftover server session {sid}")
        try:
            os.killpg(sid, 9)
        except ProcessLookupError:
            pass
    if sids:
        time.sleep(20)


def main() -> int:
    args = sys.argv[1:]
    only = None
    if args and args[0].startswith("--jobs="):  # run only these jobs, in JOBS order
        only = set(args.pop(0).split("=", 1)[1].split(","))
    for pid in (int(x) for x in args):  # jobs already running
        log(f"waiting for pid {pid}")
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(15)
        cleanup()
    for name, cmd, timeout, env, all_gpus in JOBS:
        if only is not None and name not in only:
            continue
        if all_gpus and not wait_gpus_1_3():
            log(f"{name}: SKIPPED, GPUs 1-3 still busy after 4 h")
            continue
        log(f"{name}: start (timeout {timeout} s)")
        t0 = time.monotonic()
        with open(R / f"offline_{name}.log", "w") as f:
            try:
                p = subprocess.Popen(cmd, cwd=str(HERE), env={**os.environ, **env}, stdout=f,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                rc = "TIMEOUT"
        log(f"{name}: end rc={rc} after {time.monotonic() - t0:.0f} s")
        cleanup()
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
