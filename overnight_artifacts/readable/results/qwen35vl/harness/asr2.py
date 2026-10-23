"""Two concurrent ASR eval clients (c=32 each) to test whether one client process caps throughput."""
import subprocess, sys
url, out = sys.argv[1], sys.argv[2]
PY = "/root/work/mstar/.venv/bin/python"
procs = []
for i in range(2):
    cmd = [PY, "-m", "benchmark.eval", "run", "asr", "--url", url, "--data", "/root/work/ls_clean/manifest.jsonl",
           "--concurrency", "32", "--repeats", "4", "--warmup", "32", "--out", f"{out}/c{i}", "--save-audio", "none",
           "--shuffle-seed", str(i + 1)]
    procs.append(subprocess.Popen(cmd, stdout=open(f"{out}/c{i}.log", "w"), stderr=subprocess.STDOUT))
sys.exit(max(p.wait() for p in procs))
