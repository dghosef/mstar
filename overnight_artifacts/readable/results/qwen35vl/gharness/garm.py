"""Guest ASR arm: python garm.py --tree T --arm A --size 7b --gpu 0 --port 9100 [--phase]"""
import argparse, subprocess, sys
G = "/root/guest-qwen35vl"
PY = "/root/work/mstar/.venv/bin/python"
DATA = f"{G}/ls_clean/manifest.jsonl"
def ev(conc, repeats, limit=None, warm=8):
    lim = f"--limit {limit} " if limit else ""
    return (f"{PY} -m benchmark.eval run asr --url {{url}} --data {DATA} {lim}--concurrency {conc} "
            f"--repeats {repeats} --warmup {warm} --out {{out}} --save-audio none")
ap = argparse.ArgumentParser()
for a in ("--tree", "--arm", "--size"):
    ap.add_argument(a, required=True)
ap.add_argument("--gpu", default="0"); ap.add_argument("--port", default="9100")
ap.add_argument("--phase", action="store_true"); ap.add_argument("--concs", default="1,16,64")
a = ap.parse_args()
clients = []
for c in map(int, a.concs.split(",")):
    clients.append(f"asr_c{c}::" + (ev(c, 3, limit=100) if c == 1 else ev(c, 3, warm=2 * c)))
cmd = [PY, f"{G}/gvlh/ovn.py", "run", "--tree", a.tree, "--arm", a.arm, "--serve", f"omnilingual_asr_ctc_{a.size}",
       "--config", f"configs/omnilingual_asr_ctc_{a.size}.yaml", "--out", f"/workspace/results/qwen35vl/{a.arm}",
       "--gpu", a.gpu, "--port", a.port]
if a.phase:
    cmd += ["--phase", "--phase-period", "10"]
for cl in clients:
    cmd += ["--client", cl]
sys.exit(subprocess.run(cmd).returncode)
