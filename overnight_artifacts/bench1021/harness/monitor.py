"""Mac-side overnight monitor: every INTERVAL seconds, rsync TEXT results from each bench node,
refresh the PR body, and exit (so the coordinator is woken) when a node's heartbeat goes stale
or every node reports DONE.

  python3 monitor.py            # runs until an alert; prints the reason and exits 1 (alert: stale heartbeat, ssh down, or a node idle
                                # for two checks), 2 (one node DONE: rebalance), 3 (a new line in a node's
                                # failures.jsonl), 0 (all done)
"""
import pathlib
import re
import subprocess
import sys
import time

HOME = pathlib.Path.home()
CFG = HOME / "runs/bench1021/ssh_config"
RES = HOME / "runs/bench1021/results"
LOG = HOME / "runs/bench1021/monitor.log"
NODES = {"node8": RES, "node2": RES / "diffusion"}  # host -> local results dir
INTERVAL = 900      # rsync + PR refresh + heartbeat/idle checks
FAIL_POLL = 120     # failures.jsonl check
STALE_S = 1200
NGROK = "7.tcp.ngrok.io"
TEXT = ["*.md", "*.json", "*.jsonl", "*.csv", "*.log", "*.txt", "*.cmd", "*.py"]


def log(msg):
    line = f"{time.strftime('%H:%M:%S %Z')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def ssh(host, cmd, timeout=30):
    return subprocess.run(["ssh", "-F", str(CFG), "-o", "ConnectTimeout=10", host, cmd],
                          capture_output=True, text=True, timeout=timeout)


def repin():
    """The ngrok edge IP rotates; point every Host in ssh_config at one that answers."""
    ips = subprocess.run(["dig", "+short", "@8.8.8.8", NGROK], capture_output=True, text=True).stdout.split()
    for ip in ips:
        r = subprocess.run(["ssh", "-F", str(CFG), "-o", "ConnectTimeout=8", "-o", f"HostName={ip}",
                            "node8", "true"], capture_output=True, timeout=20)
        if r.returncode == 0:
            text = CFG.read_text()
            CFG.write_text(re.sub(r"HostName \d+\.\d+\.\d+\.\d+", f"HostName {ip}", text))
            log(f"re-pinned ssh_config to {ip}")
            return True
    return False


def heartbeat(host):
    try:
        r = ssh(host, "cat /workspace/results/bench/heartbeat.txt 2>/dev/null; "
                      "stat -c %Y /workspace/results/bench/heartbeat.txt 2>/dev/null; date +%s")
    except subprocess.TimeoutExpired:
        return None, "ssh timeout"
    if r.returncode != 0:
        return None, f"ssh rc={r.returncode} {r.stderr.strip()[:120]}"
    lines = r.stdout.strip().splitlines()
    if len(lines) < 2:
        return None, "no heartbeat file yet"
    mtime, now = int(lines[-2]), int(lines[-1])
    return now - mtime, " | ".join(lines[:-2])[:200]


def gpu_busy(host):
    """True if any GPU on the node is above 5% utilization right now (None if unreadable)."""
    try:
        r = ssh(host, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    except subprocess.TimeoutExpired:
        return None
    vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    return any(v > 5 for v in vals) if vals else None


def failures(host):
    """Line count and the tail of the node's failures.jsonl (None on an ssh error)."""
    try:
        r = ssh(host, "f=/workspace/results/bench/failures.jsonl; [ -f $f ] && { wc -l < $f; tail -n 5 $f; } || echo 0")
    except subprocess.TimeoutExpired:
        return None, ""
    if r.returncode != 0 or not r.stdout.strip():
        return None, ""
    first, *rest = r.stdout.strip().splitlines()
    return int(first), "\n".join(rest)


def sync(host, dest):
    dest.mkdir(parents=True, exist_ok=True)
    inc = sum((["--include", p] for p in TEXT), [])
    subprocess.run(["rsync", "-az", "--max-size=20m", "-e", f"ssh -F {CFG}", "--include", "*/", *inc,
                    "--exclude", "*", f"{host}:/workspace/results/bench/", f"{dest}/"],
                   capture_output=True, timeout=600)


def main():
    seen_hb = {h: False for h in NODES}
    idle = {h: 0 for h in NODES}
    nfail = {h: failures(h)[0] or 0 for h in NODES}
    log(f"failures at start: {nfail}")
    while True:
        done, alerts = [], []
        for host, dest in NODES.items():
            age, info = heartbeat(host)
            if age is None and "ssh" in info and repin():
                age, info = heartbeat(host)
            log(f"{host}: age={age} {info}")
            if age is None and "ssh" in info:
                alerts.append(f"{host}: {info}")
                continue
            try:
                sync(host, dest)
            except subprocess.TimeoutExpired:
                log(f"{host}: rsync timeout")
            if age is None:  # runner not started yet; only an alert once it had been beating
                if seen_hb[host]:
                    alerts.append(f"{host}: heartbeat file vanished")
                continue
            seen_hb[host] = True
            if info.startswith("DONE"):
                done.append(host)
                continue
            if age > STALE_S:
                alerts.append(f"{host}: heartbeat stale {age}s ({info})")
            busy = gpu_busy(host)
            idle[host] = 0 if busy or busy is None else idle[host] + 1
            if idle[host] >= 2:  # two checks in a row (~15 min apart) with every GPU idle
                alerts.append(f"{host}: all GPUs idle for 2 checks ({info})")
        try:
            subprocess.run([sys.executable, str(HOME / "runs/bench1021/harness/report.py"), "--results", str(RES),
                            "--out", str(HOME / "runs/bench1021/REPORT.md"),
                            "--status", f"in progress (auto-regenerated {time.strftime('%H:%M %Z')})"],
                           capture_output=True, timeout=120, check=True)
        except Exception as e:
            log(f"REPORT.md regen failed: {e}")
        try:
            subprocess.run([sys.executable, str(HOME / "runs/overnight/harness/update_pr.py")],
                           capture_output=True, timeout=120, check=True)
        except Exception as e:  # a PR refresh failure is not worth waking for
            log(f"PR refresh failed: {e}")
        if alerts:
            log("ALERT " + "; ".join(alerts))
            sys.exit(1)
        if done and len(done) < len(NODES) and not getattr(main, "told", False):
            main.told = True
            log(f"REBALANCE {done} finished while others still run")
            sys.exit(2)
        if len(done) == len(NODES):
            log("ALL DONE")
            sys.exit(0)
        for _ in range(INTERVAL // FAIL_POLL):
            time.sleep(FAIL_POLL)
            for host in NODES:
                n, tail = failures(host)
                if n is not None and n > nfail[host]:
                    log(f"FAILURE {host}: {n - nfail[host]} new\n{tail}")
                    sys.exit(3)


if __name__ == "__main__":
    main()
