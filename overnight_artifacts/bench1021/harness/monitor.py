"""Mac-side overnight monitor: every INTERVAL seconds, rsync TEXT results from each bench node,
refresh the PR body, and exit (so the coordinator is woken) when a node's heartbeat goes stale
or every node reports DONE.

  python3 monitor.py            # runs until an alert; prints the reason and exits 1 (alert: stale heartbeat, ssh down, or a node idle
                                # for two checks), 2 (one node DONE: rebalance), 3 (a new line in a node's
                                # failures.jsonl), 0 (all done)
"""
import json
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
        # sample 10× over ~20 s; a single sample often lands between cells or during a server start
        r = ssh(host, "for i in $(seq 10); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; "
                      "sleep 2; done", timeout=60)
    except subprocess.TimeoutExpired:
        return None
    vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    return any(v > 5 for v in vals) if vals else None


def _ts(line):
    """Sortable digits of a failures.jsonl line's time (nodes write different ISO styles)."""
    try:
        t = json.loads(line).get("time", "")
    except ValueError:
        return ""
    return re.sub(r"\D", "", t)[:14]


def _tune_knob(line):
    """Expected, recorded-only failures: competitor knob sets failing in tune, and gate parity
    misses the sanity gate only annotates. Neither changes what runs."""
    try:
        r = json.loads(line)
    except ValueError:
        return False
    act = str(r.get("action_taken", ""))
    competitor_empty = "empty output" in str(r.get("error", "")) and not str(r.get("arm", "")).startswith("mstar")
    # sampled TTS occasionally never emits its end token (seen on M* and SGLang-Omni alike); recorded, not actionable
    competitor_empty = competitor_empty or "runaway length" in str(r.get("error", ""))
    # vllm-omni BAGEL edit needs 3 GPUs for image CFG; the 2-GPU lane has no vllm-omni edit arm
    competitor_empty = competitor_empty or "requires cfg_parallel_size=3" in str(r.get("error", ""))
    return (competitor_empty or str(r.get("phase", "")).startswith("tune") or "annotated" in act
            or "arm still timed" in act or "ineligible for selection" in act)


def failures(host):
    """(newest failure timestamp, its lines) from the node's failures.jsonl; (None, "") on an ssh error.
    Keyed on time, not line count: runners sometimes rewrite or reset old entries."""
    try:
        r = ssh(host, "f=/workspace/results/bench/failures.jsonl; [ -f $f ] && tail -n 200 $f || true")
    except subprocess.TimeoutExpired:
        return None, []
    if r.returncode != 0:
        return None, []
    lines = [x for x in r.stdout.splitlines() if x.strip()]
    return max((_ts(x) for x in lines), default=""), lines


def sync(host, dest):
    dest.mkdir(parents=True, exist_ok=True)
    inc = sum((["--include", p] for p in TEXT), [])
    subprocess.run(["rsync", "-az", "--max-size=20m", "-e", f"ssh -F {CFG}", "--include", "*/", *inc,
                    "--exclude", "*", f"{host}:/workspace/results/bench/", f"{dest}/"],
                   capture_output=True, timeout=600)


def main():
    seen_hb = {h: False for h in NODES}
    idle = {h: 0 for h in NODES}
    sshfail = {h: 0 for h in NODES}
    nfail = {}
    for h in NODES:  # an ssh hiccup here would make every old failure look new, so retry the baseline
        for _ in range(20):
            newest, _lines = failures(h)
            if newest is not None:
                nfail[h] = newest
                break
            time.sleep(15)
        else:
            nfail[h] = "99999999999999"
    if len(sys.argv) > 1 and sys.argv[1] == "--quiet-for":  # suppress wakes for N minutes (an acknowledged loop)
        quiet_until = time.time() + 60 * int(sys.argv[2])
    else:
        quiet_until = 0
    log(f"failures at start: {nfail}")
    while True:
        done, alerts = [], []
        for host, dest in NODES.items():
            age, info = heartbeat(host)
            if age is None and "ssh" in info and repin():
                age, info = heartbeat(host)
            log(f"{host}: age={age} {info}")
            if age is None and "ssh" in info:
                sshfail[host] += 1
                if sshfail[host] >= 2:  # one miss is usually the Mac's own network blipping
                    alerts.append(f"{host}: {info}")
                continue
            sshfail[host] = 0
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
            starting = any(k in info.lower() for k in ("starting", "server start", "setup", "compil", "cell=start"))
            busy = True if starting else gpu_busy(host)
            idle[host] = 0 if busy or busy is None else idle[host] + 1
            if idle[host] >= 3:  # three checks in a row (~45 min) with every GPU idle; phases with many server starts dip below 2
                alerts.append(f"{host}: all GPUs idle for 3 checks ({info})")
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
                newest, lines = failures(host)
                if newest is None or newest <= nfail[host]:
                    continue
                new = [x for x in lines if _ts(x) > nfail[host] and not _tune_knob(x)]
                nfail[host] = newest
                if not new:  # only competitor knob sets failing during tune: expected, recorded, not worth a wake
                    continue
                if time.time() < quiet_until:
                    continue
                log(f"FAILURE {host}: {len(new)} new\n" + "\n".join(x[:1500] for x in new[-5:]))
                sys.exit(3)


if __name__ == "__main__":
    main()
