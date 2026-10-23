"""Start/stop one mstar server per arm, for scripted A/Bs on the box.

Each arm is a tree (e.g. /root/work/base, /root/work/mstar) sharing the venv; the
server runs as ``python -m mstar.cli.main serve`` from the arm's root with the arm on
PYTHONPATH, so it imports that arm's code rather than the editable install's.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

PY = "/root/work/mstar/.venv/bin/python"
VENV_BIN = "/root/work/mstar/.venv/bin"


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def port_free(port: int) -> bool:
    out = subprocess.run(["ss", "-ltnH", f"sport = :{port}"], capture_output=True, text=True).stdout
    return not out.strip()


class Server:
    def __init__(self, arm_dir: str, args: list[str], port: int, log: str, gpus: str = "0",
                 env: dict | None = None, timeout: float = 1800):
        self.arm_dir, self.args, self.port, self.log = arm_dir, args, port, log
        self.gpus, self.timeout = gpus, timeout
        self.env = dict(os.environ)
        self.env.update({
            "PYTHONPATH": arm_dir,
            "PATH": f"{VENV_BIN}:{self.env.get('PATH', '')}",
            "VIRTUAL_ENV": "/root/work/mstar/.venv",
            "HF_HOME": "/root/work/hf",
            "MSTAR_SHM_ARENA": "1",
            "MSTAR_PHASE_TIMING": "100",
        })
        self.env.update(env or {})
        self.proc: subprocess.Popen | None = None

    def start(self) -> float:
        if not port_free(self.port):
            raise RuntimeError(f"port {self.port} busy")
        Path(self.log).parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-m", "mstar.cli.main", "serve", *self.args, "--gpus", self.gpus,
               "--port", str(self.port)]
        t0 = time.time()
        self._fh = open(self.log, "w")
        self._fh.write(f"# arm={self.arm_dir} cmd={' '.join(cmd)}\n")
        self._fh.flush()
        self.proc = subprocess.Popen(cmd, cwd=self.arm_dir, env=self.env, stdout=self._fh,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        while time.time() - t0 < self.timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server died rc={self.proc.returncode}; see {self.log}")
            if _healthy(self.port):
                return time.time() - t0
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"server not healthy after {self.timeout}s; see {self.log}")

    def stop(self) -> None:
        if self.proc is None:
            return
        pgid = self.proc.pid
        for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            t0 = time.time()
            while time.time() - t0 < wait:
                if self.proc.poll() is not None and not _group_alive(pgid):
                    break
                time.sleep(1)
            else:
                continue
            break
        self.proc = None
        self._fh.close()
        t0 = time.time()
        while not port_free(self.port) and time.time() - t0 < 30:
            time.sleep(1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def _group_alive(pgid: int) -> bool:
    out = subprocess.run(["ps", "-e", "-o", "pgid="], capture_output=True, text=True).stdout
    return any(line.strip() == str(pgid) for line in out.splitlines())


def phases(log: str) -> str:
    import sys
    sys.path.insert(0, "/root/work/mstar")
    from benchmark.worker_phases.phases import parse_log, render, segment
    return render(segment(parse_log(Path(log).read_text(errors="replace"))))
