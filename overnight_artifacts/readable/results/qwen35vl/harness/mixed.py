"""Run text_to_text and image_to_text closed loops at once against one server (a mixed workload)."""
import subprocess, sys
url, out, mid = sys.argv[1], sys.argv[2], sys.argv[3]
PY = "/root/work/mstar/.venv/bin/python"
procs = []
for rtype, tag in (("text_to_text", "t2t"), ("image_to_text", "i2t")):
    cmd = [PY, "-m", "benchmark.runner", "--url", url, "--model", "qwen3.5", "--model-id", mid, "--request-type", rtype,
           "--profiling-type", "closed_loop", "--max-concurrency", "16", "--num-requests", "480", "--num-warmup", "2",
           "--ignore-eos", "--output-len-min", "128", "--output-len-max", "128", "--output-dir", f"{out}/{tag}",
           "--local-cache", "/root/work/bcache"]
    procs.append(subprocess.Popen(cmd, stdout=open(f"{out}/{tag}.log", "w"), stderr=subprocess.STDOUT))
sys.exit(max(p.wait() for p in procs))
