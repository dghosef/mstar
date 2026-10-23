"""Run a sequence of arms: each spec is label|tree|kind|passes|env1,env2 (kind qwen:<size> or asr:<size>)."""
import subprocess, sys, time
PY = "/root/work/mstar/.venv/bin/python"
out_root = sys.argv[1]
for spec in sys.argv[2:]:
    label, tree, kind, passes, envs = (spec.split("|") + [""] * 5)[:5]
    model, size = kind.split(":")
    script = "/root/work/vlh/qwen_arm.py" if model == "qwen" else "/root/work/vlh/asr_arm.py"
    cmd = [PY, script, "--tree", tree, "--arm", label, "--size", size, "--out", f"{out_root}/{label}",
           "--passes", passes]
    for e in filter(None, envs.split(",")):
        cmd += ["--env", e]
    print(time.strftime("%H:%M:%S"), ">>> ARM", label, cmd, flush=True)
    subprocess.run(cmd)
print("CAMPAIGN DONE", flush=True)
