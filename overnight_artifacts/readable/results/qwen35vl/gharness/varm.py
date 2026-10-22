"""Guest Qwen arm: correctness (c1 greedy over images+videos+text), video c1/c8/c32, image c32, text c32."""
import argparse, subprocess, sys
G = "/root/guest-qwen35vl"; PY = "/root/work/mstar/.venv/bin/python"
ap = argparse.ArgumentParser()
for a in ("--tree", "--arm"): ap.add_argument(a, required=True)
ap.add_argument("--gpu", default="0"); ap.add_argument("--port", default="9100"); ap.add_argument("--phase", action="store_true")
ap.add_argument("--work", default="check,v1,v8,v32,i2t,t2t")
a = ap.parse_args(); w = a.work.split(",")
def runner(rtype, conc, n, outlen, warm=2):
    return (f"{PY} -m benchmark.runner --url {{url}} --model qwen3.5 --model-id Qwen/Qwen3.5-0.8B --request-type {rtype} "
            f"--profiling-type closed_loop --max-concurrency {conc} --num-requests {n} --num-warmup {warm} --ignore-eos "
            f"--output-len-min {outlen} --output-len-max {outlen} --output-dir {{out}} --local-cache {G}/bcache")
cl = []
if "check" in w: cl.append(f"check::{PY} {G}/gvlh/qcheck.py --url {{url}} --set {G}/vset --out {{out}} --conc 1")
for c in (1, 8, 32):
    if f"v{c}" in w:
        cl.append(f"video_c{c}::{PY} {G}/gvlh/qcheck.py --url {{url}} --set {G}/vset --out {{out}} --conc {c} --only t8s_360p --repeat {max(4, 2 * c)} --max-tokens 32")
if "i2t" in w: cl.append("i2t_c32::" + runner("image_to_text", 32, 640, 128))
if "t2t" in w: cl.append("t2t_c32::" + runner("text_to_text", 32, 640, 128))
cmd = [PY, f"{G}/gvlh/ovn.py", "run", "--tree", a.tree, "--arm", a.arm, "--serve", "qwen3_5_0.8b", "--config", "configs/qwen3_5_0.8b.yaml",
       "--out", f"/workspace/results/qwen35vl/{a.arm}", "--gpu", a.gpu, "--port", a.port]
if a.phase: cmd += ["--phase"]
for c in cl: cmd += ["--client", c]
sys.exit(subprocess.run(cmd).returncode)
