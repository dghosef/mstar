"""Unattended on-box queue runner for the overnight benchmark.

Driven by queue.json (re-read at every model boundary). For each model, in order, it runs
the phases below; every phase is idempotent and resumable (stage markers, and a timed cell
counts as done once results.jsonl holds it with failed == 0), so a restart continues where
it stopped. Each full pass walks the queue from the top, so an arm added later for an
earlier model (e.g. mstar-pr-cp) is picked up at the next model boundary.

Phases per model (names usable in a model's "phases" list):
  data      build sharegpt / random_1024_256 with the model's tokenizer
  correct   per arm: start (= smoke), greedy parity answers, GSM8K-250; mstar-pr twice (control)
  tune      competitors' knob sweep (random 1024/256, c = 1/32/128)
  fetune    M*'s HTTP frontend: Python vs Rust (same sweep), plus parity + GSM8K on Rust
  gate      transformers oracle (no server up) + gate.py on every collected M* tag
  rounds    3 rounds x arms x {sharegpt, random_1024_256} x c = 1..128, fresh server per round
  mtp       supplementary: vLLM / SGLang with their recipe MTP, c = 1/4/16
  xcheck    one point with our client, `vllm bench serve`, `sglang.bench_serving`
  profile   the chosen mstar-pr arm with MSTAR_PHASE_TIMING=100
  summarize summary.md + notes.txt

Decision rules (fixed before the run; each decision is logged to decisions.jsonl):
  competitor config  best geometric-mean output tok/s over the tune points (a knob set counts
                     only if all its points ran with failed == 0)
  frontend           Rust if mstar_pr_rf_a passes the gate and its geomean > Python's x 1.03,
                     else Python; the choice applies to every M* arm of that model
  gate               gate.py defaults; if mstar-pr's gate fails, the model's timed phases are
                     skipped (recorded), other models continue
  failures           every server start and cell has a hard timeout; a failed start or a cell
                     with failed > 0 is retried once on a fresh server, then recorded and the
                     runner moves on (an arm whose server never comes up is excluded for the
                     model; if mstar-pr can't come up, the model is skipped)

  python -I benchq.py        (normally via supervisor.py)
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bench", HERE / "bench.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench"] = bench
_spec.loader.exec_module(bench)
from bench import CLIENT_PY, PORT, TREES, TUNE_POINTS, TUNING, WORK, WORKLOADS, Bench  # noqa: E402

QUEUE = HERE / "queue.json"
RELOAD_FLAG = HERE / ".reload"     # touch it: the runner restarts itself at the next phase boundary
RELOAD_RC = 3                      # the supervisor restarts on this code without logging a failure
SOURCES = [HERE / f for f in ("benchq.py", "bench.py", "client.py")]
RESULTS = bench.RESULTS
HEART = RESULTS / "heartbeat.txt"
DECISIONS = RESULTS / "decisions.jsonl"
FAILURES = RESULTS / "failures.jsonl"
ALL_PHASES = ["setup", "data", "smoke", "correct", "tune", "fetune", "gate", "rounds", "mtp", "xcheck", "profile", "summarize"]
STATE = {"pos": "-", "model": "-", "phase": "-", "arm": "-", "cell": "-"}
CELL_TIMEOUT = 1800


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def heartbeat():
    while True:
        try:
            HEART.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} runner pid {os.getpid()} "
                             f"model {STATE['pos']} {STATE['model']} phase={STATE['phase']} arm={STATE['arm']} "
                             f"cell={STATE['cell']}\n")
        except OSError:
            pass
        time.sleep(30)


def state(**kw):
    STATE.update({k: str(v) for k, v in kw.items()})


def decide(model, what, choice, **numbers):
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": model, "decision": what,
           "choice": choice, **numbers}
    with DECISIONS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"DECISION {model} {what}: {choice} {numbers}")


def tail(path, n=20):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def failure(model, arm, phase, cell, error, action):
    """A final failure (after its retry): one line in failures.jsonl, watched from the Mac."""
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": model, "arm": arm,
           "phase": phase, "cell": cell, "error": str(error)[-4000:], "action_taken": action}
    with FAILURES.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"FAILURE {model} {arm} {phase} {cell}: {action}")


def gpu_pids():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def cleanup():
    """Between phases: stop our server and kill anything still holding a GPU."""
    subprocess.run([str(HERE / "server.sh"), "stop", "bench"], capture_output=True, timeout=300)
    for _ in range(3):
        pids = gpu_pids()
        if not pids:
            return
        log(f"cleanup: killing leftover GPU processes {pids}")
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(10)


def run_sub(cmd, timeout, **kw) -> int:
    kw.setdefault("cwd", "/root/work")   # never the harness dir: its files must not shadow imports
    try:
        return subprocess.run(cmd, timeout=timeout, **kw).returncode
    except subprocess.TimeoutExpired:
        log(f"timeout after {timeout}s: {cmd[:4]}")
        return -9


class ModelRun:
    def __init__(self, spec: dict, extra_arms: list[dict]):
        self.spec = spec
        self.key = spec["model"]
        self.id = spec.get("id", self.key)
        self.b = Bench(self.key)
        self.out = self.b.out
        self.arms = list(spec.get("arms", ["mstar-main", "vllm", "sglang", "mstar-pr"]))
        for ea in extra_arms:
            if self.key in ea.get("models", []):
                TREES[ea["arm"]] = Path(ea["tree"])
                TREES[ea["arm"] + "-rf"] = Path(ea["tree"])
                if ea["arm"] not in self.arms:
                    self.arms.append(ea["arm"])
        self.start_timeout = spec.get("start_timeout")
        self.excluded = self._excluded()
        self.kind = self.b.kind
        self.tune_wl = spec.get("tune_workload", "random_1024_256")
        self.tune_points = [tuple(x) for x in spec.get("tune_points", TUNE_POINTS)]
        self.competitors = [a for a in self.arms if a not in TREES and not a.endswith("-rf")]
        p = self.out / "stages" / "smoke_skip.json"
        self.smoke_skip = {tuple(x) for x in json.loads(p.read_text())} if p.exists() else set()

    # ---- helpers ---------------------------------------------------------------------
    def marker(self, name):
        return self.out / "stages" / f"{name}.done"

    def _excluded(self):
        return {p.stem[len("excluded_"):] for p in (self.out / "stages").glob("excluded_*.done")}

    def exclude(self, arm, why):
        self.marker(f"excluded_{arm}").touch()
        self.excluded.add(arm)
        decide(self.key, f"exclude {arm}", "excluded", reason=why)
        failure(self.id, arm, STATE["phase"], "-", why,
                "arm excluded for this model" + ("; model skipped" if arm == "mstar-pr" else ""))

    def frontend(self):
        return self.b.tuned().get("mstar_frontend") or self.spec.get("frontend") or "python"

    def served(self, arm):
        """The arm name a timed phase serves: M* arms on the chosen frontend."""
        if arm in TREES and not arm.endswith("-rf") and self.frontend() == "rust":
            return arm + "-rf"
        return arm

    def start(self, arm, label, extra=None, env_extra=None) -> bool:
        cleanup()
        for attempt in (1, 2):
            state(arm=arm, cell=f"starting {label} (attempt {attempt})")
            if self.b.start(arm, label if attempt == 1 else f"{label}_retry", extra, env_extra,
                            timeout=self.start_timeout):
                try:
                    self.b.smoke()
                    return True
                except Exception as e:  # noqa: BLE001
                    log(f"!! smoke {label}: {e!r}")
                    self.b.stop()
            cleanup()
        srv = self.out / "servers" / f"{label}_retry.log"
        failure(self.id, arm, STATE["phase"], label, tail(srv), "server did not start twice; see action of caller")
        return False

    def parity_collect(self, tag, prompts, pdir) -> int:
        pdir.mkdir(exist_ok=True)
        return run_sub([str(CLIENT_PY), "-I", str(HERE / "parity.py"), "collect", "--port", str(PORT), "--tag", tag,
                        "--model-id", self.b.m["hf"], "--prompts", str(prompts), "--out-dir", str(pdir)], 1800)

    def collect(self, tag, gsm8k=True, arm=None) -> bool:
        """The correctness answers for one server (kind-specific), while it is up."""
        env = {**os.environ, "HF_HOME": str(WORK / "hf")}
        url = f"http://127.0.0.1:{PORT}"
        if self.kind == "text":
            rc = self.parity_collect(tag, HERE / "prompts_text.jsonl", self.out / "parity")
            if rc == 0 and gsm8k:
                rc = run_sub([str(CLIENT_PY), "-I", str(HERE / "gsm8k.py"), "--url", url,
                              "--model-id", self.b.m["hf"], "--n", "250", "--conc", "32",
                              "--out", str(self.out / f"gsm8k_{tag}.json")], 1800, env=env)
            return rc == 0
        if self.kind == "vision":
            rc = self.parity_collect(tag, self.b.data / "prompts_vision_img.jsonl", self.out / "parity")
            rv = self.parity_collect(tag, self.b.data / "prompts_vision_video.jsonl", self.out / "parity_video")
            if rv != 0:
                failure(self.id, tag, "correct", "video parity", "video parity collection failed",
                        "video answers missing for this arm; image gate still applies")
            if rc == 0 and gsm8k:
                rc = run_sub([str(CLIENT_PY), "-I", str(HERE / "mcq_eval.py"), "--url", url, "--model-id",
                              self.b.m["hf"], "--rows", str(self.b.data / "mmmu.jsonl"), "--conc", "16",
                              "--out", str(self.out / f"mmmu_{tag}.json")], 1800, env=env)
            return rc == 0
        # omni: GSM8K (text out), LibriSpeech A2T WER, and T2S speech kept for the Whisper check
        text_only = json.dumps({"modalities": ["text"]})
        rc = run_sub([str(CLIENT_PY), "-I", str(HERE / "gsm8k.py"), "--url", url, "--model-id", self.b.m["hf"],
                      "--n", "250", "--conc", "32", "--extra-body", text_only,
                      "--out", str(self.out / f"gsm8k_{tag}.json")], 2400, env=env)
        d = self.out / "gate_a2t" / tag
        ra = run_sub([str(CLIENT_PY), "-I", str(HERE / "client.py"), "--url", url, "--model-id", self.b.m["hf"],
                      "--data", str(self.b.data / "omni_a2t.jsonl"), "--conc", "16", "--num", "200",
                      "--warmup-waves", "1", "--warmup-offset", "0", "--tokenizer", self.b.m["hf"], "--out", str(d)]
                     + bench.WL_CLIENT["omni_a2t"], 2400, env={**env, "HF_HUB_OFFLINE": "1"})
        if ra == 0:
            run_sub([str(WORK / "mstar-main/.venv/bin/python"), "-I", str(HERE / "wer.py"), "a2t", "--results",
                     str(d / "results.json"), "--rows", str(self.b.data / "omni_a2t.jsonl"),
                     "--out", str(self.out / f"wer_a2t_{tag}.json")], 600, env=env)
        self.b.cell_runner("gate", arm, "gate", 4, 1, 1, num=32, out_dir=self.out / "gate_t2s" / tag)
        return rc == 0 and ra == 0

    def do_cell(self, tag, arm, config, wl, c, rnd, warm, num=None):
        if wl == "omni_t2s":
            return self.b.cell_runner(tag, arm, config, c, rnd, warm, num=num, timeout=CELL_TIMEOUT)
        return self.b.cell(tag, arm, config, wl, c, rnd, warm, num=num, timeout=CELL_TIMEOUT)

    def ph_smoke(self):
        """Every arm starts and answers a few requests of every workload before anything is
        timed. An arm whose server can't start is excluded; a workload an arm can't serve is
        skipped for that arm (both recorded in failures.jsonl)."""
        if self.kind == "text" and not self.spec.get("smoke"):
            return   # text models smoke inside their correctness phase
        wls = list(self.spec.get("workloads", WORKLOADS))
        for arm in self.arms:
            if arm in self.excluded or self.marker(f"smoke_{arm}").exists():
                continue
            state(phase="smoke", arm=arm)
            cfg, args = self.b.arm_args(arm)
            if not self.start(arm, f"smoke_{arm}", args):
                self.exclude(arm, "server did not start (twice) in the smoke phase")
                continue
            self.b.versions(arm, f"smoke_{arm}")
            bad = []
            for wl in wls:
                state(cell=f"smoke {wl}")
                row = self.do_cell("smoke", arm, cfg, wl, 2, 1, 0, num=4)
                if row.get("failed") != 0 or row["notes"]:
                    bad.append(wl)
                    self.smoke_skip.add((arm, wl))
                    failure(self.id, arm, "smoke", wl, f"failed={row.get('failed')} errors={row.get('errors')} "
                            f"notes={row['notes']}\n" + tail(row.get("log", "")),
                            "workload skipped for this arm (and its -rf variant)")
            self.b.stop()
            (self.out / "stages" / "smoke_skip.json").write_text(json.dumps(sorted(self.smoke_skip)))
            if bad and len(bad) == len(wls):
                self.exclude(arm, f"every workload failed its smoke requests: {bad}")
            self.marker(f"smoke_{arm}").touch()

    def good_cells(self, tag):
        return {(r["arm"], r["workload"], r["conc"], r["round"]) for r in self.b.rows()
                if r["tag"] == tag and r.get("failed") == 0 and r.get("output_tok_s") and not r["notes"]}

    def run_cells(self, tag, arm, config, extra, rnd, cells, label):
        """cells: [(wl, c)] not yet done. One server; a failed cell is retried once on a fresh one."""
        if not cells:
            return
        if not self.start(arm, label, extra):
            decide(self.key, f"{tag} {arm} r{rnd}", "skipped: server did not come up twice")
            failure(self.id, arm, tag, f"r{rnd}", tail(self.out / "servers" / f"{label}_retry.log"),
                    "round skipped for this arm")
            return
        self.b.versions(arm, label)
        first = True
        for wl, c in cells:
            state(arm=arm, cell=f"{tag} {wl} c={c} r{rnd}")
            row = self.do_cell(tag, arm, config, wl, c, rnd, 5 if first else 1)
            first = False
            if row.get("failed") == 0 and not row["notes"]:
                continue
            log(f"!! cell failed ({row.get('failed')} {row.get('errors')} {row['notes']}); retrying on a fresh server")
            self.b.stop()
            if not self.start(arm, label + f"_retry_{wl}_c{c}", extra):
                decide(self.key, f"{tag} {arm} r{rnd}", "rest of round skipped: restart failed",
                       at=f"{wl} c={c}")
                failure(self.id, arm, tag, f"{wl} c={c} r{rnd}", tail(row.get("log", "")),
                        "rest of the round skipped (restart after a failed cell failed)")
                return
            row = self.do_cell(tag, arm, config, wl, c, rnd, 5)
            if row.get("failed") != 0 or row["notes"]:
                decide(self.key, f"cell {tag} {arm} {wl} c={c} r{rnd}", "failed twice; recorded",
                       failed=row.get("failed"), errors=row.get("errors"))
                failure(self.id, arm, tag, f"{wl} c={c} r{rnd}",
                        f"failed={row.get('failed')} errors={row.get('errors')} notes={row['notes']}\n"
                        + tail(row.get("log", "")), "recorded as failed; moved to the next cell")
        self.b.stop()

    # ---- phases ----------------------------------------------------------------------
    def ph_setup(self):
        """Idempotent: installs whatever this model needs that's missing on this node (M* trees'
        venvs, competitor venvs, client env, weights, datasets). Trees themselves are rsynced in."""
        for a in self.arms:
            tree = TREES.get(a)
            if tree and not (tree / "pyproject.toml").exists():
                self.exclude(a, f"tree {tree} missing on this node")
        trees = " ".join(t for t in ("mstar", "mstar-main") if (WORK / t / "pyproject.toml").exists())
        rc = run_sub(["bash", str(HERE / "provision.sh")], 7200,
                     env={**os.environ, "MODELS": self.b.m["hf"], "TREES_TO_SETUP": trees},
                     stdout=(self.out / "setup.log").open("a"), stderr=subprocess.STDOUT)
        if rc != 0:
            decide(self.key, "setup", f"provision.sh rc={rc}; continuing")
        for a in self.arms:   # extra trees (e.g. mstar-pr-cp) bring their own venv
            tree = TREES.get(a)
            if tree and a not in self.excluded and not (tree / ".venv/bin/python").exists():
                rc = run_sub(["bash", str(HERE / "setup_box.sh"), str(tree)], 3600,
                             stdout=(self.out / "setup.log").open("a"), stderr=subprocess.STDOUT)
                decide(self.key, f"setup venv {tree}", "ok" if rc == 0 else f"failed rc={rc}")

    def ph_data(self):
        env = {**os.environ, "HF_HOME": str(WORK / "hf")}
        py = str(WORK / "mstar-main/.venv/bin/python")   # datasets, PIL, soundfile; ffmpeg on PATH
        if self.kind == "vision":
            if not (self.b.data / "prompts_vision_video.jsonl").exists():
                state(phase="data", cell="vision_data")
                rc = run_sub([py, "-I", str(HERE / "vision_data.py"), "--out", str(self.b.data)], 5400, env=env,
                             stdout=(self.out / "data.log").open("a"), stderr=subprocess.STDOUT)
                if rc != 0:
                    failure(self.id, "-", "data", "vision_data", tail(self.out / "data.log"), "vision data incomplete")
            return
        if self.kind == "omni":
            if not (self.b.data / "omni_t2t.jsonl").exists():
                state(phase="data", cell="omni_data")
                sg = RESULTS / "q35_0.8b" / "data" / "sharegpt.jsonl"
                rc = run_sub([py, "-I", str(HERE / "omni_data.py"), "--out", str(self.b.data), "--sharegpt", str(sg)],
                             3600, env=env, stdout=(self.out / "data.log").open("a"), stderr=subprocess.STDOUT)
                if rc != 0:
                    failure(self.id, "-", "data", "omni_data", tail(self.out / "data.log"), "omni data incomplete")
            return
        wls = sorted(set(self.spec.get("workloads", WORKLOADS)) | set(WORKLOADS))
        if any(not (self.b.data / f"{w}.jsonl").exists() for w in wls):
            state(phase="data", cell=",".join(wls))
            run_sub([str(bench.ENGINES / "vllm/bin/python"), "-I", str(HERE / "make_data.py"), "--tokenizer",
                     self.b.m["hf"], "--sharegpt", str(bench.SHAREGPT), "--out", str(self.b.data),
                     "--workloads", ",".join(wls)], 3600,
                    env={**os.environ, "HF_HOME": str(WORK / "hf"), "HF_HUB_OFFLINE": "1"})

    def ph_correct(self):
        for arm in self.arms:
            if arm in self.excluded:
                continue
            for rep in (["a", "b"] if arm == "mstar-pr" and self.kind != "omni" else ["a"]):
                tag = f"{arm}_{rep}".replace("-", "_")
                if self.marker(f"correct_{tag}").exists():
                    continue
                state(phase="correct", arm=arm)
                cfg, args = self.b.arm_args(arm)
                if not self.start(arm, f"correct_{tag}", args):
                    self.exclude(arm, "server did not start (twice) in the correctness phase")
                    break
                self.b.versions(arm, f"correct_{tag}")
                ok = self.collect(tag, gsm8k=(rep == "a"), arm=arm)
                self.b.stop()
                if ok:
                    self.marker(f"correct_{tag}").touch()
                else:
                    decide(self.key, f"correct {tag}", "collection failed; continuing without it")
                    failure(self.id, arm, "correct", tag, tail(self.out / "servers" / f"correct_{tag}.log"),
                            "parity/GSM8K collection failed; continuing without it")
                    self.marker(f"correct_{tag}").touch()

    def ph_tune(self):
        knobs = self.spec.get("tune_knobs", "all")
        if knobs == []:
            if not self.marker("tune_none").exists():
                decide(self.key, "competitor tuning", "none: each engine runs its own recommended default config",
                       reason=self.spec.get("tune_note", ""))
                self.marker("tune_none").touch()
            return
        if isinstance(knobs, dict) and "from" in knobs:   # default + another model's winners
            src = json.loads((RESULTS / knobs["from"] / "tuned.json").read_text())
            knobs = ["default"] + [src[e]["config"] for e in ("vllm", "sglang") if e in src]
        for eng in ("vllm", "sglang"):
            if eng not in self.arms or eng in self.excluded:
                continue
            for name, args in TUNING[eng]:
                if knobs != "all" and name not in knobs:
                    continue
                label = f"tune_{eng}_{name}"
                if self.marker(label).exists():
                    continue
                state(phase="tune", arm=f"{eng}[{name}]")
                if not self.start(eng, label, args):
                    with (self.out / "tune_failed.jsonl").open("a") as f:
                        f.write(json.dumps({"engine": eng, "knob": name, "args": args}) + "\n")
                    self.marker(label).touch()
                    continue
                first = True
                for c, n in self.tune_points:
                    state(cell=f"tune {self.tune_wl} c={c}")
                    self.b.cell("tune", eng, name, self.tune_wl, c, 1, 3 if first else 1, num=n,
                                timeout=CELL_TIMEOUT)
                    first = False
                self.b.stop()
                self.marker(label).touch()
        before = self.b.tuned()
        bench.TUNE_POINTS = tuple(self.tune_points)   # pick_tuned counts a knob only if all points ran
        self.b.pick_tuned()
        after = self.b.tuned()
        for eng in ("vllm", "sglang"):
            if eng in after and before.get(eng, {}).get("config") != after[eng]["config"]:
                decide(self.key, f"{eng} config", after[eng]["config"], geomean_tok_s=after[eng]["geomean_tok_s"])

    def ph_fetune(self):
        if self.spec.get("frontend"):
            t = self.b.tuned()
            if t.get("mstar_frontend") != self.spec["frontend"]:
                t["mstar_frontend"] = self.spec["frontend"]
                t["mstar_frontend_reason"] = "fixed in queue.json: " + self.spec.get("frontend_note", "")
                (self.out / "tuned.json").write_text(json.dumps(t, indent=1))
                decide(self.key, "frontend", self.spec["frontend"], reason=t["mstar_frontend_reason"])
            return
        for arm in ("mstar-pr", "mstar-pr-rf"):
            label = f"tune_{arm}"
            if self.marker(label).exists():
                continue
            state(phase="fetune", arm=arm)
            if not self.start(arm, label):
                with (self.out / "tune_failed.jsonl").open("a") as f:
                    f.write(json.dumps({"engine": arm, "knob": "frontend", "args": []}) + "\n")
                self.marker(label).touch()
                continue
            self.b.versions(arm, label)
            if arm.endswith("-rf"):
                self.collect("mstar_pr_rf_a", arm=arm)
            cfgname = "rust-frontend" if arm.endswith("-rf") else "python-frontend"
            first = True
            for c, n in self.tune_points:
                state(cell=f"fetune {self.tune_wl} c={c}")
                self.b.cell("tune", arm, cfgname, self.tune_wl, c, 1, 3 if first else 1, num=n,
                            timeout=CELL_TIMEOUT)
                first = False
            self.b.stop()
            self.marker(label).touch()

    def ph_gate(self):
        if self.kind == "vision":
            return self.gate_vision()
        if self.kind == "omni":
            return self.gate_omni()
        return self.gate_text()

    def run_oracle(self, prompts, pdir, gate_name):
        tags = sorted(f.stem for f in pdir.glob("*.json") if not f.stem.startswith("report"))
        stamp = self.out / "stages" / f"{gate_name}_tags.json"
        if not (stamp.exists() and json.loads(stamp.read_text()) == tags and (self.out / gate_name).exists()):
            state(phase="gate", arm=f"oracle {gate_name}")
            cleanup()
            self.b.phase_oracle(prompts, pdir, gate_name)
            stamp.write_text(json.dumps(tags))
        txt = (self.out / gate_name).read_text() if (self.out / gate_name).exists() else ""
        v = {}
        for blk in txt.split("#### gate ")[1:]:
            v[blk.split("\n", 1)[0].strip()] = "PASS" if "GATE PASS" in blk else "FAIL"
        return v, txt, (pdir / "report.json").exists()

    def record_verdicts(self, verdicts, detail):
        prev = json.loads((self.out / "stages" / "gate_verdicts.json").read_text()) \
            if (self.out / "stages" / "gate_verdicts.json").exists() else {}
        for k, v in verdicts.items():
            if prev.get(k) != v:
                decide(self.key, f"gate {k}", v, detail=detail.get(k, ""))
                if v == "FAIL":
                    failure(self.id, k, "gate", k, detail.get(k, ""),
                            "timed phases skipped for this model" if k == "mstar_pr_a" else "recorded; arm still timed")
        (self.out / "stages" / "gate_verdicts.json").write_text(json.dumps(verdicts))
        (self.out / "gate_summary.txt").write_text("".join(f"{k}: {v}  {detail.get(k, '')}\n"
                                                           for k, v in verdicts.items()))

    def gate_vision(self):
        """Fixed before the run: an M* tag passes iff (1) its image-prompt parity passes gate.py,
        (2) its video-prompt parity passes too (when the video oracle ran; otherwise UNCHECKED,
        recorded, not blocking), and (3) its MMMU-200 accuracy >= min(vLLM, SGLang) - 0.05."""
        vi, _, _ = self.run_oracle(self.b.data / "prompts_vision_img.jsonl", self.out / "parity", "gate.txt")
        vv, _, ran = self.run_oracle(self.b.data / "prompts_vision_video.jsonl", self.out / "parity_video",
                                     "gate_video.txt")
        if not ran and not self.marker("video_oracle_missing").exists():
            failure(self.id, "-", "gate", "video oracle", tail(self.out / "oracle_gate_video.log", 30),
                    "video parity UNCHECKED (oracle did not run); image + MMMU gates still apply")
            self.marker("video_oracle_missing").touch()

        def acc(tag):
            f = self.out / f"mmmu_{tag}.json"
            return json.loads(f.read_text())["accuracy"] if f.exists() else None
        comp = [a for a in (acc("vllm_a"), acc("sglang_a")) if a is not None]
        floor = min(comp) - 0.05 if comp else None
        verdicts, detail = {}, {}
        for tag in vi:
            a = acc(tag)
            mm_ok = floor is None or (a is not None and a >= floor)
            v_ok = vv.get(tag, "UNCHECKED") != "FAIL"
            verdicts[tag] = "PASS" if vi[tag] == "PASS" and v_ok and mm_ok else "FAIL"
            detail[tag] = (f"image parity {vi[tag]}; video parity {vv.get(tag, 'UNCHECKED')}; MMMU {a} "
                           f"(floor {floor and round(floor, 3)} = min(vLLM, SGLang) - 0.05)")
        self.record_verdicts(verdicts, detail)
        self.frontend_rule(verdicts)
        return verdicts

    def gate_omni(self):
        """Fixed before the run, per M* tag vs the two competitors (vllm-omni, sglang-omni):
        GSM8K-250 (text out) >= min(competitors) - 0.03; LibriSpeech test-clean A2T WER <=
        min(competitors) + 0.01; T2S Whisper WER of the speech vs the engine's own text <=
        min(competitors) + 0.02, and no silent WAV."""
        gdir = self.out / "gate_t2s"
        for d in (sorted(gdir.glob("*")) if gdir.exists() else []):
            out = self.out / f"wer_t2s_{d.name}.json"
            if not out.exists() and any(d.glob("req_*.wav")):
                state(phase="gate", arm=f"whisper {d.name}")
                cleanup()
                run_sub([str(WORK / "mstar-main/.venv/bin/python"), "-I", str(HERE / "wer.py"), "t2s", "--dir", str(d),
                         "--out", str(out)], 3600, env={**os.environ, "HF_HOME": str(WORK / "hf"),
                                                        "CUDA_VISIBLE_DEVICES": "0"})

        def val(prefix, tag, key):
            f = self.out / f"{prefix}_{tag}.json"
            try:
                return json.loads(f.read_text())[key]
            except (OSError, ValueError, KeyError):
                return None
        comps = [f"{c.replace('-', '_')}_a" for c in self.competitors if c not in self.excluded]
        verdicts, detail = {}, {}
        for arm in self.arms:
            if arm not in TREES or arm in self.excluded:
                continue
            tag = f"{arm}_a".replace("-", "_")
            g, a2t, t2s = val("gsm8k", tag, "accuracy"), val("wer_a2t", tag, "wer"), val("wer_t2s", tag, "wer")
            sil = val("wer_t2s", tag, "silent")
            cg = [x for x in (val("gsm8k", c, "accuracy") for c in comps) if x is not None]
            ca = [x for x in (val("wer_a2t", c, "wer") for c in comps) if x is not None]
            ct = [x for x in (val("wer_t2s", c, "wer") for c in comps) if x is not None]
            ok_g = g is not None and (not cg or g >= min(cg) - 0.03)
            ok_a = a2t is not None and (not ca or a2t <= min(ca) + 0.01)
            ok_t = t2s is not None and not sil and (not ct or t2s <= min(ct) + 0.02)
            verdicts[tag] = "PASS" if ok_g and ok_a and ok_t else "FAIL"
            detail[tag] = (f"GSM8K {g} (competitors {cg}); A2T WER {a2t} (competitors {ca}); "
                           f"T2S WER {t2s}, silent {sil} (competitors {ct})")
        self.record_verdicts(verdicts, detail)
        return verdicts

    def gate_text(self):
        pdir = self.out / "parity"
        tags = sorted(f.stem for f in pdir.glob("*.json") if not f.stem.startswith("report"))
        stamp = self.out / "stages" / "gate_tags.json"
        if stamp.exists() and json.loads(stamp.read_text()) == tags and (self.out / "gate.txt").exists():
            pass
        else:
            state(phase="gate", arm="oracle")
            cleanup()
            self.b.phase_oracle()
            stamp.write_text(json.dumps(tags))
        gate = (self.out / "gate.txt").read_text() if (self.out / "gate.txt").exists() else ""
        verdicts = {}
        for blk in gate.split("#### gate ")[1:]:
            name = blk.split("\n", 1)[0].strip()
            verdicts[name] = "PASS" if "GATE PASS" in blk else "FAIL"
        prev = json.loads((self.out / "stages" / "gate_verdicts.json").read_text()) \
            if (self.out / "stages" / "gate_verdicts.json").exists() else {}
        for k, v in verdicts.items():
            if prev.get(k) != v:
                gs = self.out / f"gsm8k_{k}.json"
                acc = json.loads(gs.read_text())["accuracy"] if gs.exists() else None
                decide(self.key, f"gate {k}", v, gsm8k=acc)
                if v == "FAIL":
                    blk = gate.split(f"#### gate {k}")[1].split("####")[0]
                    failure(self.id, k, "gate", k, blk.strip(),
                            "timed phases skipped for this model" if k == "mstar_pr_a" else "recorded; arm still timed")
        (self.out / "stages" / "gate_verdicts.json").write_text(json.dumps(verdicts))
        self.frontend_rule(verdicts)
        return verdicts

    def frontend_rule(self, verdicts):
        """Rust frontend iff its correctness gate passed and its tune geomean > Python's x 1.03."""
        t = self.b.tuned()
        if not self.spec.get("frontend") and "mstar_frontend" not in t:
            rows = [r for r in self.b.rows() if r["tag"] == "tune" and r["workload"] == self.tune_wl]
            score = {}
            for arm in ("mstar-pr", "mstar-pr-rf"):
                ok = [r for r in rows if r["arm"] == arm and r.get("failed") == 0 and r.get("output_tok_s")]
                if len(ok) == len(self.tune_points):
                    score[arm] = math.exp(statistics.mean(math.log(r["output_tok_s"]) for r in ok))
            py, rs = score.get("mstar-pr"), score.get("mstar-pr-rf")
            rust_ok = verdicts.get("mstar_pr_rf_a") == "PASS"
            choice = "rust" if (rust_ok and py and rs and rs > py * 1.03) else "python"
            t["mstar_frontend"] = choice
            t["mstar_frontend_geomean"] = score
            t["mstar_frontend_reason"] = (f"python {py and round(py)} vs rust {rs and round(rs)} geomean tok/s; "
                                          f"rust gate {verdicts.get('mstar_pr_rf_a', 'missing')}; "
                                          "rule: rust if its gate passes and > +3%")
            (self.out / "tuned.json").write_text(json.dumps(t, indent=1))
            decide(self.key, "frontend", choice, python_geomean=py, rust_geomean=rs,
                   rust_gate=verdicts.get("mstar_pr_rf_a"))

    def gate_ok(self) -> bool:
        v = json.loads((self.out / "stages" / "gate_verdicts.json").read_text()) \
            if (self.out / "stages" / "gate_verdicts.json").exists() else {}
        return v.get("mstar_pr_a") == "PASS"

    def ph_rounds(self):
        if not self.gate_ok():
            if not self.marker("timed_skipped").exists():
                decide(self.key, "timed phases", "skipped: mstar-pr gate did not pass (or missing)")
                failure(self.id, "mstar-pr", "rounds", "-", tail(self.out / "gate.txt", 30),
                        "model's timed phases skipped (gate not passed)")
                self.marker("timed_skipped").touch()
            return
        concs = tuple(self.spec.get("concs", bench.CONCS))
        wls = tuple(self.spec.get("workloads", WORKLOADS))
        arms = [a for a in self.arms if a not in self.excluded]
        if self.frontend() == "rust":   # no server is up here: building is not concurrent with timing
            for a in arms:
                tree = TREES.get(a)
                if tree and not (tree / bench.RUST_BIN).exists():
                    state(phase="build rust frontend", arm=a)
                    run_sub(["bash", "-c", f"cd {tree} && . ~/.cargo/env && cargo build --release "
                             "--manifest-path rust/server/Cargo.toml"], 1200)
                    decide(self.key, f"build rust frontend {a}", "built" if (tree / bench.RUST_BIN).exists()
                           else "build failed: arm will fail to start")
        rotate = [a for a in arms if self.spec.get("rotate_pr", True) or not a.startswith("mstar-pr")]
        last = [a for a in arms if a not in rotate]
        for rnd in self.spec.get("rounds", [1, 2, 3]):
            k = (rnd - 1) % max(1, len(rotate))
            order = rotate[k:] + rotate[:k] + last
            for base in order:
                arm = self.served(base)
                done = self.good_cells("main")
                attempts = [r for r in self.b.rows() if r["tag"] == "main" and r["arm"] == arm and r["round"] == rnd]
                todo = [(wl, c) for wl in wls for c in concs if (arm, wl, c, rnd) not in done
                        and (base, wl) not in self.smoke_skip
                        and sum(1 for r in attempts if (r["workload"], r["conc"]) == (wl, c)) < 2]
                if not todo:
                    continue
                state(phase=f"rounds r{rnd}", arm=arm)
                config, args = self.b.arm_args(base)
                self.run_cells("main", arm, config, args, rnd, todo, f"main_{arm}_r{rnd}")
                self.b.summarize(quiet=True)

    def ph_mtp(self):
        if not self.gate_ok() or self.kind != "text":
            return
        concs = tuple(self.spec.get("mtp_concs", [1, 4, 16]))
        for rnd in self.spec.get("mtp_rounds", [1, 2]):
            for eng in ("vllm", "sglang"):
                if eng not in self.arms or eng in self.excluded:
                    continue
                done = self.good_cells("mtp")
                todo = [(wl, c) for wl in WORKLOADS for c in concs if (eng, wl, c, rnd) not in done]
                attempts = [r for r in self.b.rows() if r["tag"] == "mtp" and r["arm"] == eng and r["round"] == rnd]
                todo = [x for x in todo if sum(1 for r in attempts if (r["workload"], r["conc"]) == x) < 2]
                if not todo:
                    continue
                state(phase=f"mtp r{rnd}", arm=eng)
                config, args = self.b.arm_args(eng)
                tag = f"{eng}_mtp"
                if not (self.out / "parity" / f"{tag}.json").exists():
                    if self.start(eng, f"mtp_{eng}_gate", args + Bench.MTP[eng]):
                        self.collect(tag)
                        self.b.stop()
                self.run_cells("mtp", eng, config + "+mtp", args + Bench.MTP[eng], rnd, todo, f"mtp_{eng}_r{rnd}")

    def ph_xcheck(self):
        if not self.gate_ok() or self.kind != "text":
            return
        arms = [self.served("mstar-pr"), self.served("mstar-main"), "vllm", "sglang"]
        state(phase="xcheck")
        cleanup()
        self.b.phase_xcheck([a for a in arms if a.replace("-rf", "") not in self.excluded])

    def ph_profile(self):
        if not self.gate_ok():
            return
        arm = self.served("mstar-pr")
        if self.marker(f"profile_{arm}").exists():
            return
        state(phase="profile", arm=arm)
        cleanup()
        pts = self.spec.get("profile_points", ["random_1024_256:128", "random_1024_256:32", "sharegpt:128", "sharegpt:1"])
        try:
            self.b.phase_profile([(p.split(":")[0], int(p.split(":")[1])) for p in pts], arm)
        except Exception as e:  # noqa: BLE001
            log(f"!! profile: {e!r}")
        self.marker(f"profile_{arm}").touch()

    def ph_summarize(self):
        state(phase="summarize")
        self.b.summarize(quiet=True)
        t = self.b.tuned()
        verd = json.loads((self.out / "stages" / "gate_verdicts.json").read_text()) \
            if (self.out / "stages" / "gate_verdicts.json").exists() else {}
        decs = [json.loads(x) for x in DECISIONS.read_text().splitlines() if x.strip()] if DECISIONS.exists() else []
        decs = [d for d in decs if d["model"] == self.key]
        gs = {g.stem[6:]: json.loads(g.read_text())["accuracy"] for g in sorted(self.out.glob("gsm8k_*.json"))}
        lines = [f"{self.key}: {self.b.m['hf']} on GPUs {self.b.m['gpus']}",
                 f"arms: {', '.join(self.served(a) for a in self.arms if a not in self.excluded)}"
                 + (f"; excluded: {sorted(self.excluded)}" if self.excluded else ""),
                 f"competitor configs: " + ", ".join(f"{e}={t[e]['config']} {t[e]['args']}" for e in ("vllm", "sglang") if e in t),
                 f"M* frontend: {t.get('mstar_frontend', '?')} ({t.get('mstar_frontend_reason', '')})",
                 "gate: " + ", ".join(f"{k} {v}" for k, v in verd.items()),
                 "GSM8K: " + ", ".join(f"{k} {v:.3f}" for k, v in gs.items()),
                 f"decisions logged: {len(decs)} (decisions.jsonl)"]
        (self.out / "notes.txt").write_text("\n".join(lines) + "\n")

    def done_marker(self):
        return self.marker("model_complete" if self.id == self.key else f"entry_complete_{self.id}")

    def complete(self) -> bool:
        m = self.done_marker()
        return m.exists() and json.loads(m.read_text() or "[]") == sorted(self.arms)

    def run(self):
        phases = self.spec.get("phases", ALL_PHASES)
        for ph in phases:
            if RELOAD_FLAG.exists():
                RELOAD_FLAG.unlink()
                log("reload requested: exiting for the supervisor to restart (phase boundary)")
                cleanup()
                sys.exit(RELOAD_RC)
            state(model=self.key, phase=ph, arm="-", cell="-")
            log(f"==== {self.key} phase {ph}")
            if "mstar-pr" in self.excluded and ph != "summarize":
                continue
            try:
                getattr(self, f"ph_{ph}")()
            except Exception:  # noqa: BLE001 -- a phase crashing must not stop the queue
                log(f"!! phase {ph} crashed:\n{traceback.format_exc()}")
                decide(self.key, f"phase {ph}", "crashed; moving on", tb=traceback.format_exc()[-600:])
                failure(self.id, STATE["arm"], ph, STATE["cell"], traceback.format_exc()[-3000:],
                        "phase abandoned; continuing with the next phase")
            finally:
                cleanup()
        self.done_marker().write_text(json.dumps(sorted(self.arms)))


def main():
    threading.Thread(target=heartbeat, daemon=True).start()
    q = json.loads(QUEUE.read_text())
    sa = q.get("start_after")
    while sa and not (Path(sa["log"]).exists() and sa["text"] in Path(sa["log"]).read_text()):
        state(model="waiting", phase=f"start_after {Path(sa['log']).name}")
        time.sleep(20)
    stamp = {str(f): f.stat().st_mtime for f in SOURCES}
    cleanup()
    # one-off setup steps (installs, downloads), run with no server up, before any model
    for step in q.get("setup", []):
        mk = RESULTS / "setup_stages" / f"{step['name']}.done"
        if mk.exists():
            continue
        if step.get("only_for") and not any(m["model"] in step["only_for"] for m in q["models"]):
            continue
        mk.parent.mkdir(parents=True, exist_ok=True)
        state(model="setup", phase=step["name"], arm="-", cell="-")
        log(f"==== setup {step['name']}")
        with (RESULTS / "setup_stages" / f"{step['name']}.log").open("w") as f:
            rc = run_sub(["bash", "-c", step["cmd"]], step.get("timeout", 3600), stdout=f, stderr=subprocess.STDOUT)
        decide("setup", step["name"], "ok" if rc == 0 else f"failed rc={rc}")
        if rc == 0:
            mk.touch()
        else:   # not marked done: the next runner start (and each model's setup phase) retries it
            failure("setup", "-", "setup", step["name"], tail(RESULTS / "setup_stages" / f"{step['name']}.log"),
                    "setup step failed and stays pending; retried at the next runner start / model setup")
    while True:
        q = json.loads(QUEUE.read_text())   # re-read at every model boundary
        todo = None
        for i, spec in enumerate(q["models"]):
            mr = ModelRun(spec, q.get("extra_arms", []))
            if not mr.complete():
                todo = mr
                state(pos=f"{i + 1}/{len(q['models'])}")
                break
        if todo is None:
            state(model="DONE", phase="-", arm="-", cell="-")
            log("QUEUE DONE")
            return 0
        if {str(f): f.stat().st_mtime for f in SOURCES} != stamp:
            log("harness sources changed: exiting for the supervisor to restart (model boundary)")
            cleanup()
            return RELOAD_RC
        log(f"######## model {STATE['pos']} {todo.id} arms {todo.arms}")
        todo.run()


if __name__ == "__main__":
    sys.exit(main())
