"""One Qwen3.5 arm: a throughput server pass and a phase-timing pass.

    python qwen_arm.py --tree /root/work/base --arm base --size 0.8 --out /workspace/results/r1/base
        [--passes perf,phase] [--conc 32] [--env K=V]
"""
import argparse
import subprocess
import sys

H = "/root/work/vlh"
PY = "/root/work/mstar/.venv/bin/python"


def runner(model_id, rtype, conc, n, outlen, warm=3):
    return (f"{PY} -m benchmark.runner --url {{url}} --model qwen3.5 --model-id {model_id} "
            f"--request-type {rtype} --profiling-type closed_loop --max-concurrency {conc} "
            f"--num-requests {n} --num-warmup {warm} --ignore-eos --output-len-min {outlen} "
            f"--output-len-max {outlen} --output-dir {{out}} --local-cache /root/work/bcache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--size", default="0.8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", default="perf,phase")
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--workloads", default="check,check8,i2t,t2t,i2t_c1,big,huge,xl,phone,mixed")
    a = ap.parse_args()
    mid = f"Qwen/Qwen3.5-{a.size}B"
    serve = f"qwen3_5_{a.size}b"
    cfg = f"configs/qwen3_5_{a.size}b.yaml"
    wl = a.workloads.split(",")
    c = a.conc
    perf_clients = []
    if "check" in wl:
        perf_clients.append(f"check::{PY} {H}/qcheck.py --url {{url}} --set /root/work/imgset --out {{out}} --conc 1")
    if "i2t" in wl:
        perf_clients.append(f"i2t_c{c}::" + runner(mid, "image_to_text", c, 30 * c, 128))
    if "t2t" in wl:
        perf_clients.append(f"t2t_c{c}::" + runner(mid, "text_to_text", c, 20 * c, 128))
    if "check8" in wl:
        perf_clients.append(f"check_c8::{PY} {H}/qcheck.py --url {{url}} --set /root/work/imgset --out {{out}} "
                            "--conc 8 --repeat 4")
    if "i2t_c1" in wl:
        perf_clients.append("i2t_c1::" + runner(mid, "image_to_text", 1, 40, 32, warm=5))
    if "big" in wl:
        perf_clients.append(f"big_c8::{PY} {H}/qcheck.py --url {{url}} --set /root/work/imgset --out {{out}} "
                            "--conc 8 --only big --repeat 8 --max-tokens 32")
    for kind in ("huge", "xl", "phone"):
        if kind in wl:
            perf_clients.append(f"{kind}_c4::{PY} {H}/qcheck.py --url {{url}} --set /root/work/imgset --out {{out}} "
                                f"--conc 4 --only {kind} --repeat 6 --max-tokens 32")
    if "mixed" in wl:
        perf_clients.append(f"mixed::{PY} {H}/mixed.py {{url}} {{out}} {mid}")
    phase_clients = [f"i2t_c{c}_phase::" + runner(mid, "image_to_text", c, 20 * c, 128, warm=2),
                     f"t2t_c{c}_phase::" + runner(mid, "text_to_text", c, 6 * c, 128, warm=2)]
    envs = sum((["--env", e] for e in a.env), [])
    rc = 0
    for p in a.passes.split(","):
        clients = perf_clients if p == "perf" else phase_clients
        cmd = [PY, f"{H}/ovn.py", "run", "--tree", a.tree, "--arm", a.arm, "--serve", serve,
               "--config", cfg, "--out", f"{a.out}/{p}", *envs]
        if p == "phase":
            cmd.append("--phase")
        for cl in clients:
            cmd += ["--client", cl]
        print(">>>", p, flush=True)
        rc |= subprocess.run(cmd).returncode
    print("ARM DONE", rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
