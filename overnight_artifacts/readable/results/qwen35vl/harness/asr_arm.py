"""One omnilingual ASR CTC arm: perf pass (c=1, c=16, c=64 with WER) and phase pass.

    python asr_arm.py --tree /root/work/base --arm base --size 300m --out DIR [--passes perf,phase]
"""
import argparse
import subprocess
import sys

H = "/root/work/vlh"
PY = "/root/work/mstar/.venv/bin/python"
DATA = "/root/work/ls_clean/manifest.jsonl"


def ev(conc, repeats, limit=None, warm=8):
    lim = f"--limit {limit} " if limit else ""
    return (f"{PY} -m benchmark.eval run asr --url {{url}} --data {DATA} {lim}--concurrency {conc} "
            f"--repeats {repeats} --warmup {warm} --out {{out}} --save-audio none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--size", default="300m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", default="perf,phase")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--concs", default="1,16,64")
    ap.add_argument("--pyspy", action="store_true")
    ap.add_argument("--phase-c1", action="store_true")
    ap.add_argument("--two-clients", action="store_true", default=True)
    a = ap.parse_args()
    serve = f"omnilingual_asr_ctc_{a.size}"
    cfg = f"configs/omnilingual_asr_ctc_{a.size}.yaml"
    perf = []
    for c in a.concs.split(","):
        c = int(c)
        perf.append(f"asr_c{c}::" + (ev(c, 2, limit=100) if c == 1 else ev(c, 3, warm=2 * c)))
    if a.two_clients:
        perf.append(f"asr_2x32::{PY} {H}/asr2.py {{url}} {{out}}")
    phase = ["asr_c64_phase::" + ev(64, 8, warm=64)]
    if a.phase_c1:
        phase = ["asr_c1_phase::" + ev(1, 2, limit=100)]
    envs = sum((["--env", e] for e in a.env), [])
    rc = 0
    for p in a.passes.split(","):
        clients = perf if p == "perf" else phase
        cmd = [PY, f"{H}/ovn.py", "run", "--tree", a.tree, "--arm", a.arm, "--serve", serve,
               "--config", cfg, "--out", f"{a.out}/{p}", *envs]
        if p == "phase":
            cmd += ["--phase", "--phase-period", "10"]
            cmd += ["--pyspy", "asr_c1_phase" if a.phase_c1 else "asr_c64_phase"]
        for cl in clients:
            cmd += ["--client", cl]
        print(">>>", p, flush=True)
        rc |= subprocess.run(cmd).returncode
    print("ARM DONE", rc, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
