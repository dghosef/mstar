#!/usr/bin/env python3
"""Generic entries of the node2 queue (dqueue.py loads this per phase for entries with "module": "dgen"):
M* (PR d248477ec, main abdfd5802) vs each model's strongest available competitor, for models the
diffusion (dbench) and TTS (dtts) harnesses don't cover: ASR (Higgs Audio STT, Omnilingual ASR), BAGEL,
Cosmos3, LingBot-World, V-JEPA2, Waypoint. Model table: dgen_models.py (GEN); research notes and the
reason for every competitor choice live there too.

Arms: mstar-pr and mstar-main (each tree's own config), plus per model either
  - a served competitor (an HTTP server, timed with the same closed-loop client, dgen_client.py), or
  - an offline competitor: the official package's own batch inference in its own venv (one process per
    workload; batch b plays the role of concurrency c), labelled "offline batch" everywhere it is
    reported, because it has no server, queueing or HTTP cost.
A model with no runnable competitor is reported as mstar-pr vs mstar-main only (said in its summary).

Phases: setup (idempotent: weights, items, competitor installs + one-request smoke; an engine whose
install or smoke fails is dropped and logged), tune (competitor knob sets at the tune points, each also
collecting its gate outputs), collect (M* PR twice = same-code control, M* main; untimed gate outputs),
gate (dgen_gate.py, sanity criteria fixed in its docstring), select, rounds, profile, summarize.

Per server and round: smoke, an untimed warm pass, then per point one warmup wave + the measured
requests; a failed cell is retried once on a fresh server; 3 rounds, arm order rotated per round; a
point counts only with failed == 0. Metrics: requests/s, units/s (audio-s/s = RTFx for ASR, frames/s
for video), latency mean/p50/p99, TTFB p50/p99.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

import dbench
from dbench import CLIENT_PY, DIFF_PY, ENG, HERE, PORT0, RESULTS, STATUS, TREES, WORK, failure, log, loadavg, \
    reap_all, tail

sys.path.insert(0, str(HERE))
import dgen_models  # noqa: E402

dgen_models = importlib.reload(dgen_models)
GEN = dgen_models.GEN
DATA = WORK / "data/gen"
GATE_GPU = "7"
VOSP = ENG / "vllmomni/lib/python3.12/site-packages/vllm_omni"


def ctx_of(g, port=PORT0):
    """Placeholders usable in a model table's commands, env and install steps."""
    return {"port": port, "port2": port + 100, "eng": str(ENG), "work": str(WORK), "here": str(HERE),
            "hf": g.get("hf_id", ""), "vosp": str(VOSP), "pr": str(TREES["mstar-pr"]), "main": str(TREES["mstar-main"])}


def model_cfg(spec):
    g = json.loads(json.dumps(GEN[spec.get("model", spec["key"])]))
    for k, v in spec.get("overrides", {}).items():
        g[k] = v
    return g


class GenBench(dbench.Bench):
    def __init__(self, spec):
        self.spec = spec
        self.g = model_cfg(spec)
        key = spec["key"]
        dbench.MODELS.setdefault(key, {"kind": self.g["kind"], "hf": self.g.get("hf_id", key), "steps": None})
        super().__init__(key)
        DATA.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ arms and knobs
    def dropped(self):
        p = self.out / "dropped.json"
        d = json.loads(p.read_text()) if p.exists() else {}
        if self.g.get("no_main"):
            d.setdefault("mstar-main", self.g["no_main"])
        return d

    def comps(self):
        d = self.dropped()
        return [c for c in self.g.get("comps", {}) if c not in d]

    def arms(self):
        d = self.dropped()
        return [a for a in ("mstar-pr", "mstar-main") if a not in d] + self.comps()

    def comp(self, arm):
        return self.g["comps"][arm]

    def offline(self, arm):
        return arm not in TREES and self.comp(arm)["type"] == "offline"

    def knob_table(self, arm):
        if arm in TREES:
            return self.g.get("mstar_knobs") or {"config": []}
        return self.comp(arm).get("knobs") or {"default": []}

    def knobs_of(self, arm, name=None):
        table = self.knob_table(arm)
        if name is None:
            key = "mstar_knob" if arm in TREES else arm
            name = self.tuned().get(key, {}).get("config") or next(iter(table))
        return name, table[name]

    def workloads(self):
        return list(self.g["workloads"])

    def supports(self, arm, wl):
        w = self.g["workloads"][wl]
        if arm in TREES:
            return "mstar" in w["spec"]
        return arm in w["spec"] or (self.offline(arm) and wl in self.comp(arm).get("workloads", [wl]))

    def items(self, wl):
        return DATA / self.g.get("data_key", self.key) / f"{wl}.jsonl"

    def label(self, arm):
        if arm == "mstar-pr":
            return "M* PR"
        if arm == "mstar-main":
            return "M* main"
        c = self.comp(arm)
        return c.get("label", arm) + (" (offline batch)" if c["type"] == "offline" else "")

    # ------------------------------------------------------------ servers
    def env_base(self):
        env = dict(os.environ)
        env.update(HF_HOME=str(WORK / "hf"), PATH=f"/usr/local/bin:{env['PATH']}")
        tok = WORK / "hf/token"
        if tok.exists():
            env["HF_TOKEN"] = tok.read_text().strip()
        return env

    def gpus(self, arm):
        if arm in TREES:
            return self.g.get("gpus", "0")
        return self.comp(arm).get("gpus", "0")

    def server_cmd(self, arm, gpu, port, args, env_extra=None):
        env = self.env_base()
        if arm in TREES:
            tr = TREES[arm]
            env["PATH"] = f"{tr}/.venv/bin:{env['PATH']}"
            env.update(MSTAR_SHM_ARENA="1", TORCHINDUCTOR_CACHE_DIR=str(WORK / f"inductor/{arm}"))
            env.update({k: v.format(**ctx_of(self.g, port)) for k, v in self.g.get("mstar_env", {}).items()})
            cfg = self.g["config"] if arm == "mstar-pr" else self.g.get("config_main", self.g["config"])
            cmd = [f"{tr}/.venv/bin/python", "-m", "mstar.cli.main", "serve", self.g["mstar"], "--config",
                   str(tr / cfg), "--gpus", self.gpus(arm), "--port", str(port)] + list(args)
            cwd = tr
        else:
            c = self.comp(arm)
            ctx = ctx_of(self.g, port)
            cmd = [x.format(**ctx) for x in c["cmd"]] + [x.format(**ctx) for x in args]
            env.update(CUDA_VISIBLE_DEVICES=self.gpus(arm), TORCHINDUCTOR_CACHE_DIR=str(WORK / f"inductor/{arm}"))
            env.update({k: v.format(**ctx) for k, v in c.get("env", {}).items()})
            if c.get("path_prepend"):
                env["PATH"] = c["path_prepend"].format(**ctx) + ":" + env["PATH"]
            cwd = c.get("cwd", str(ENG)).format(**ctx)
        if env_extra:
            env.update(env_extra)
        return cmd, env, cwd

    def health_path(self, arm):
        return "/health" if arm in TREES else self.comp(arm).get("health", "/health")

    def start(self, arm, label_, gpu=0, port=PORT0, args=(), env_extra=None, name=None):
        name = name or f"g{port}"
        cmd, env, cwd = self.server_cmd(arm, gpu, port, args, env_extra)
        slog = self.out / "servers" / f"{label_}.log"
        (self.out / "servers" / f"{label_}.cmd").write_text(shlex.join(cmd) + "\n" + json.dumps(
            {k: env[k] for k in env if k.startswith(("MSTAR_", "CUDA_VISIBLE", "TORCHINDUCTOR", "VLLM", "SGLANG"))})
            + "\n")
        limit = self.g.get("start_timeout", 3600 if arm in TREES else 2400)
        log(f"start {label_} gpus {self.gpus(arm)}:{port}: {shlex.join(cmd)}")
        t0 = time.time()
        try:
            r = subprocess.run([str(dbench.SERVER_SH), "start", name, str(port), "--health", self.health_path(arm),
                                "--timeout", str(limit), "--log", str(slog), "--", *cmd], env=env, cwd=cwd,
                               capture_output=True, text=True, timeout=limit + 120)
            ok, err = r.returncode == 0, r.stderr[-500:]
        except subprocess.TimeoutExpired:
            ok, err = False, "start timed out"
        secs = round(time.time() - t0, 1)
        with (self.out / "startup.jsonl").open("a") as f:
            f.write(json.dumps({"label": label_, "arm": arm, "seconds": secs, "ok": ok, "time": time.time()}) + "\n")
        if not ok:
            log(f"!! {label_}: start failed after {secs}s: {err}")
            self.stop(name)
            return None
        log(f"   {label_} healthy after {secs}s")
        self.live = {"name": name, "port": port, "log": str(slog), "health": self.health_path(arm)}
        return name

    # ------------------------------------------------------------ clients
    def req_spec(self, arm, wl):
        s = self.g["workloads"][wl]["spec"]
        return s["mstar"] if arm in TREES else s[arm]

    def client(self, arm, wl, out, conc, num, warm, seed_base=1000, save=None):
        w = self.g["workloads"][wl]
        cmd = [str(CLIENT_PY), "-I", str(HERE / "dgen_client.py"), "--engine", arm, "--port", str(PORT0),
               "--spec", json.dumps(self.req_spec(arm, wl)), "--items", str(self.items(wl)), "--conc", str(conc),
               "--num", str(num), "--warmup-waves", str(warm), "--seed-base", str(seed_base),
               "--timeout", str(w.get("req_timeout", 1800)), "--out", str(out)]
        if save:
            cmd += ["--save-dir", str(save)]
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "results.json").unlink(missing_ok=True)
        per = w.get("per_req_s", 60)
        limit = int(600 + per * math.ceil((num + warm * conc) / conc) * (1 + conc / 8))
        return self.watched_client(cmd, out, limit, idle_s=900 if warm == 0 else 300)

    def offline_run(self, arm, wl, out, batches, nums, warm, seed_base, kargs, save=None):
        """One process of the competitor's own batch inference: loads once, runs every batch size."""
        c = self.comp(arm)
        ctx = ctx_of(self.g)
        py = c["python"].format(**ctx)
        cmd = [py, "-I", str(HERE / c["script"]), "--workload", wl, "--items", str(self.items(wl)),
               "--batch-sizes", ",".join(map(str, batches)), "--nums", ",".join(map(str, nums)),
               "--warmup", str(warm), "--seed-base", str(seed_base), "--out", str(out)] + \
            [x.format(**ctx) for x in c.get("args", [])] + [x.format(**ctx) for x in kargs]
        if save:
            cmd += ["--save-dir", str(save)]
        env = self.env_base()
        env.update(CUDA_VISIBLE_DEVICES=self.gpus(arm), TORCHINDUCTOR_CACHE_DIR=str(WORK / f"inductor/{arm}"))
        env.update({k: v.format(**ctx) for k, v in c.get("env", {}).items()})
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "results.json").unlink(missing_ok=True)
        lp = Path(str(out) + ".log")
        log(f"offline {arm} {wl} b={batches}: {shlex.join(cmd)}")
        per = self.g["workloads"][wl].get("per_req_s", 60)
        limit = int(c.get("load_s", 1800) + per * sum(math.ceil(n / b) + warm for b, n in zip(batches, nums)) * 2)
        with open(lp, "w") as f:
            rc, reason = dbench.run_watched(cmd, limit, {"log": str(lp), "idle_s": 1200}, env=env,
                                            cwd=c.get("cwd", str(ENG)).format(**ctx), stdout=f,
                                            stderr=subprocess.STDOUT)
        try:
            rows = json.loads((Path(out) / "results.json").read_text())["rows"]
        except (OSError, ValueError, KeyError):
            rows = [{"batch": b, "num": n, "completed": 0, "failed": None,
                     "error": f"rc={rc} {reason or ''} {tail(lp, 5)}"[:1500]} for b, n in zip(batches, nums)]
        return rows, lp

    # ------------------------------------------------------------ passes
    def done_keys(self, tag):
        return {(r["arm"], r.get("knob"), r.get("workload"), r["conc"], r["round"]) for r in self.rows()
                if r.get("tag") == tag and r.get("failed") == 0 and "conc" in r}

    def points(self, arm, wls=None, tune=False):
        pts = []
        for wl in wls or self.workloads():
            if not self.supports(arm, wl):
                continue
            w = self.g["workloads"][wl]
            if tune:
                pts += [(wl, c, n) for c, n in w.get("tune", [])]
            else:
                concs = w["concs"]
                if not (arm in TREES) and self.offline(arm):
                    concs = w.get("offline_batches", concs)
                pts += [(wl, c, w["nums"][str(c)]) for c in concs]
        return pts

    def row(self, tag, arm, kname, wl, c, rnd, n, s, la0, t0, offline=False):
        return {"tag": tag, "arm": arm, "knob": kname, "workload": wl, "conc": c, "round": rnd, "steps": None,
                "num": n, "load_before": la0, "load_after": loadavg(), "started": t0,
                "seconds": round(time.time() - t0, 1), "offline_batch": offline, **s}

    def gen_pass(self, arm, rnd, tag, knob=None, points=None, gate_tag=None):
        """One server process (or one offline process per workload): smoke, warm pass, the cells not yet
        done, then (if asked) this arm's gate outputs. A failed pass is retried once on a fresh process."""
        kname, kargs = self.knobs_of(arm, knob)
        points = self.points(arm) if points is None else points
        last_err, last_log, last_cell = "", None, "server start"
        for attempt in (1, 2):
            done = self.done_keys(tag)
            todo = [(wl, c, n) for wl, c, n in points if (arm, kname, wl, c, rnd) not in done]
            gate_todo = [] if gate_tag is None else [
                wl for wl in self.workloads() if self.g["workloads"][wl].get("gate_n") and self.supports(arm, wl)
                and not (self.out / "gate" / gate_tag / wl / "DONE").exists()]
            if not todo and not gate_todo:
                return True
            STATUS.update(phase=tag, arm=f"{arm}[{kname}] r{rnd} attempt {attempt}", cell="start")
            reap_all("before gen pass")
            lab = f"{tag}_{arm}_{kname}_r{rnd}" + ("" if attempt == 1 else "_retry")
            if self.offline(arm):
                ok, last_err, last_log = self.offline_pass(arm, kname, kargs, rnd, tag, todo, gate_tag, gate_todo, lab)
                reap_all("after offline pass")
                if ok:
                    return True
                if attempt == 1:
                    self.decision("retry", tag=tag, arm=arm, knob=kname, round=rnd)
                continue
            last_log = self.out / "servers" / f"{lab}.log"
            name = self.start(arm, lab, 0, PORT0, kargs)
            if not name:
                last_err = "server did not become healthy"
                continue
            try:
                wl0 = (todo or [(gate_todo[0], 1, 1)])[0][0]
                s = self.client(arm, wl0, self.out / "servers" / f"smoke_{lab}", 1, 1, 0, seed_base=7)
                if s.get("failed") != 0:
                    raise RuntimeError(f"smoke failed: {s}")
                ok = True
                for wl in dict.fromkeys(p[0] for p in todo):
                    w = self.g["workloads"][wl]
                    STATUS.update(cell=f"warm pass {wl}")
                    wc = min(w.get("warm_conc", 8), max(c for x, c, _ in todo if x == wl))
                    self.client(arm, wl, self.out / "cells" / tag / arm / f"warmpass_{wl}_r{rnd}_{kname}", wc,
                                w.get("warm_n", 2 * wc), 0, seed_base=900000)
                    for _, c, n in [p for p in todo if p[0] == wl]:
                        last_cell = f"{wl} c={c}"
                        STATUS.update(cell=f"{tag} {arm}[{kname}] {wl} c={c} r{rnd}")
                        la0, t0 = loadavg(), time.time()
                        d = self.out / "cells" / tag / arm / f"{wl}_c{c}_r{rnd}_{kname}"
                        s = self.client(arm, wl, d, c, n, 1, seed_base=1000 + 10000 * rnd)
                        self.append_row(self.row(tag, arm, kname, wl, c, rnd, n, s, la0, t0))
                        log(f"   {tag} {arm}[{kname}] {wl} c={c} r{rnd}: {self.fmt(s)} failed={s.get('failed')}")
                        if s.get("failed") != 0:
                            ok = False
                            last_err = f"cell failed={s.get('failed')} {s.get('errors') or s.get('error')}"
                            break
                    if not ok:
                        break
                if ok:
                    for wl in gate_todo:
                        self.collect_gate(arm, wl, gate_tag)
                    (self.out / "versions").mkdir(exist_ok=True)
                    (self.out / "versions" / f"{arm}.json").write_text(json.dumps(self.gversions(arm)))
                    return True
            except Exception as e:  # noqa: BLE001
                last_err = f"pass exception: {e!r}"
                log(f"!! {tag} {arm}: {e!r}")
            finally:
                self.stop(name)
                reap_all("after gen pass")
            if "illegal memory access" in last_err + str(getattr(self, "last_abort", "")):
                self.mark_knob_failed(arm, kname, "illegal memory access")
                break
            if attempt == 1:
                self.decision("retry", tag=tag, arm=arm, knob=kname, round=rnd)
        self.decision("pass_failed", tag=tag, arm=arm, knob=kname, round=rnd)
        failure(self.key, f"{arm}[{kname}]", tag, f"r{rnd} {last_cell}",
                f"{last_err}\n--- log tail ---\n{tail(last_log) if last_log else ''}",
                "recorded as failed after one retry on a fresh process; queue moves on")
        return False

    def collect_gate(self, arm, wl, gate_tag):
        w = self.g["workloads"][wl]
        g = self.out / "gate" / gate_tag / wl
        STATUS.update(cell=f"gate collect {gate_tag} {wl}")
        s = self.client(arm, wl, g, w.get("gate_conc", 4), w["gate_n"], 0, seed_base=1000, save=g / "out")
        (g / "DONE").write_text(json.dumps({"failed": s.get("failed"), "completed": s.get("completed")}))
        log(f"   gate collect {gate_tag} {wl}: failed={s.get('failed')}")

    def offline_pass(self, arm, kname, kargs, rnd, tag, todo, gate_tag, gate_todo, lab):
        ok, err, lp = True, "", None
        for wl in dict.fromkeys([p[0] for p in todo] + list(gate_todo)):
            pts = [(c, n) for x, c, n in todo if x == wl]
            if pts:
                STATUS.update(cell=f"{tag} {arm}[{kname}] {wl} b={[c for c, _ in pts]} r{rnd}")
                la0, t0 = loadavg(), time.time()
                d = self.out / "cells" / tag / arm / f"{wl}_r{rnd}_{kname}" / lab
                rows, lp = self.offline_run(arm, wl, d, [c for c, _ in pts], [n for _, n in pts], 1,
                                            1000 + 10000 * rnd, kargs)
                for r in rows:
                    s = {k: v for k, v in r.items() if k not in ("batch",)}
                    self.append_row(self.row(tag, arm, kname, wl, r["batch"], rnd, r.get("num"), s, la0, t0, True))
                    log(f"   {tag} {arm}[{kname}] {wl} b={r['batch']} r{rnd} (offline): {self.fmt(r)} "
                        f"failed={r.get('failed')}")
                    if r.get("failed") != 0:
                        ok, err = False, f"batch {r['batch']} failed={r.get('failed')} {r.get('error')}"
            if wl in gate_todo and ok:
                w = self.g["workloads"][wl]
                g = self.out / "gate" / gate_tag / wl
                rows, lp = self.offline_run(arm, wl, g, [w.get("gate_conc", 4)], [w["gate_n"]], 0, 1000, kargs,
                                            save=g / "out")
                f = rows[0].get("failed") if rows else None
                (g / "DONE").write_text(json.dumps({"failed": f, "completed": rows[0].get("completed") if rows else 0}))
        if ok:
            (self.out / "versions").mkdir(exist_ok=True)
            (self.out / "versions" / f"{arm}.json").write_text(json.dumps(self.gversions(arm)))
        return ok, err, lp

    def fmt(self, s):
        u = self.g.get("units", "units")
        v, r, l50 = s.get("units_per_s"), s.get("throughput_per_s"), s.get("lat_p50")
        return (f"{v and round(v, 3)} {u}/s, {r and round(r, 3)} req/s, lat p50 {l50 and round(l50, 3)}s, "
                f"ttfb p50 {s.get('ttfb_p50') and round(s['ttfb_p50'], 3)}s")

    def gversions(self, arm):
        if arm in TREES:
            return self.versions(arm)
        c = self.comp(arm)
        out = {"label": c.get("label", arm), "note": c.get("note", "")}
        venv = c.get("venv")
        if venv:
            r = subprocess.run(["uv", "pip", "freeze", "--python", str(ENG / venv / "bin/python")], capture_output=True,
                               text=True, env={**os.environ, "PATH": f"/usr/local/bin:{os.environ['PATH']}"})
            keep = tuple(c.get("version_pkgs", ())) + ("torch=", "transformers", "vllm", "sglang", "flash", "triton")
            out["packages"] = [x for x in r.stdout.splitlines() if x.lower().startswith(keep)]
        if c.get("git"):
            r = subprocess.run(["git", "-C", c["git"].format(eng=ENG), "describe", "--tags", "--always"],
                               capture_output=True, text=True)
            out["git"] = r.stdout.strip()
        return out


# ---------------------------------------------------------------- setup
def setup(b):
    env = b.env_base()
    subprocess.run(["bash", str(HERE / "install_engines.sh")], env=env, timeout=7200,
                   stdout=open(b.out / "setup.log", "a"), stderr=subprocess.STDOUT)
    # weights + items (dgen_data.py builds the per-workload items files)
    repos = b.g.get("weights", [])
    if repos:
        code = "from huggingface_hub import snapshot_download as s\nfor r in %r: print(s(r))" % repos
        r = subprocess.run([str(TREES["mstar-pr"] / ".venv/bin/python"), "-c", code], env=env, capture_output=True,
                           text=True, timeout=7200)
        if r.returncode != 0:
            raise RuntimeError(f"weights download failed: {r.stderr[-600:]}")
    for step in b.g.get("pre_setup", []):  # e.g. weights that M* fetches from a non-HF URL
        p = subprocess.run(["bash", "-c", step.format(**ctx_of(b.g))], env=env, timeout=7200,
                           capture_output=True, text=True)
        (b.out / "setup_pre.log").open("a").write(p.stdout + p.stderr)
        if p.returncode != 0:
            raise RuntimeError(f"pre-setup step failed: {step[:200]}: {(p.stdout + p.stderr)[-600:]}")
    for wl in b.workloads():
        if not b.items(wl).exists():
            b.items(wl).parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.run([str(TREES["mstar-pr"] / ".venv/bin/python"), "-I", str(HERE / "dgen_data.py"),
                                "--builder", b.g["workloads"][wl]["items"], "--out", str(b.items(wl))], env=env,
                               timeout=3600, capture_output=True, text=True)
            (b.out / "setup_data.log").open("a").write(p.stdout + p.stderr)
            # the builder renames its output into place only when complete; some dataset streamers abort at
            # interpreter exit after that, so the file (not the exit code) is the success signal
            if not b.items(wl).exists():
                raise RuntimeError(f"items for {wl} not built (rc={p.returncode}): {(p.stdout + p.stderr)[-600:]}")
    dropped = b.dropped()
    for eng, c in b.g.get("comps", {}).items():
        if eng in dropped or not c.get("install"):
            continue
        p = subprocess.run(["bash", "-c", "set -uo pipefail\nexport PATH=/usr/local/bin:$PATH HF_HOME=/root/work/hf\n"
                            + c["install"].format(**ctx_of(b.g))], env=env, timeout=7200,
                           capture_output=True, text=True, cwd=str(ENG))
        (b.out / f"setup_{eng}.log").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            dropped[eng] = f"install failed rc={p.returncode}"
            failure(b.key, eng, "setup", "install", (p.stdout + p.stderr)[-3000:], "engine dropped from this model")
    (b.out / "dropped.json").write_text(json.dumps(dropped, indent=1))
    # one-request smoke per arm on GPU 0: an engine that cannot serve one request is dropped
    for arm in b.comps() + [a for a in ("mstar-pr", "mstar-main") if a not in dropped]:
        STATUS.update(phase="setup smoke", arm=arm)
        reap_all("before smoke")
        kname, kargs = b.knobs_of(arm, next(iter(b.knob_table(arm))))
        wl = next(w for w in b.workloads() if b.supports(arm, w))
        ok, why = False, ""
        if b.offline(arm):
            rows, lp = b.offline_run(arm, wl, b.out / "servers" / f"setup_smoke_{arm}", [1], [1], 0, 7, kargs)
            ok = bool(rows) and rows[0].get("failed") == 0
            why = tail(lp, 30)
        else:
            name = b.start(arm, f"setup_smoke_{arm}", 0, PORT0, kargs)
            if name:
                try:
                    s = b.client(arm, wl, b.out / "servers" / f"setup_smoke_{arm}_client", 1, 2, 0, seed_base=7)
                    ok = s.get("failed") == 0
                    why = json.dumps(s)[:1500]
                    log(f"setup smoke {arm}: {s}")
                finally:
                    b.stop(name)
            else:
                why = tail(b.out / "servers" / f"setup_smoke_{arm}.log", 30)
        reap_all("after smoke")
        if not ok:
            if arm == "mstar-pr":
                failure(b.key, arm, "setup", "smoke", why, "M* PR smoke failed; the model's timed phases will record failures")
            else:
                dropped[arm] = "smoke failed"
                failure(b.key, arm, "setup", "smoke", why, "arm dropped from this model (setup smoke failed)")
    (b.out / "dropped.json").write_text(json.dumps(dropped, indent=1))
    b.decision("gen_arms", arms=b.arms(), dropped=dropped)


# ---------------------------------------------------------------- tune / collect / gate / select
def tune(b):
    for arm in b.comps():
        table = b.knob_table(arm)
        for k in table:
            if b.marker(f"tune_{arm}_{k}").exists():
                continue
            # the tune points (if any) and this knob set's gate outputs, on one process
            b.gen_pass(arm, 0, f"tune_{k}", k, b.points(arm, tune=True), gate_tag=f"{arm}__{k}")
            b.marker(f"tune_{arm}_{k}").touch()
    res = {}
    for r in b.rows():
        if r.get("tag", "").startswith("tune_") and "conc" in r:
            res.setdefault(r["arm"], {}).setdefault(r["knob"], {})[f"{r['workload']}:{r['conc']}"] = r
    out = {}
    for arm, sets in res.items():
        for k, rs in sets.items():
            ok = rs and all(x.get("failed") == 0 and x.get("units_per_s") for x in rs.values())
            out.setdefault(arm, {})[k] = {
                "score": math.exp(statistics.fmean(math.log(x["units_per_s"]) for x in rs.values())) if ok else None,
                "per_point": {p: x.get("units_per_s") for p, x in rs.items()}}
    (b.out / "tune.json").write_text(json.dumps(out, indent=1))


def collect(b):
    """M* PR twice (two processes: the same-code control) and M* main; gate outputs only (untimed)."""
    for arm, tag in (("mstar-pr", "mstar_pr"), ("mstar-pr", "mstar_pr_b"), ("mstar-main", "mstar_main")):
        if arm in b.dropped():
            continue
        b.gen_pass(arm, 0, f"collect_{tag}", None, [], gate_tag=tag)


def gate(b):
    g = b.out / "gate"
    tags = sorted(d.name for d in g.iterdir() if d.is_dir())
    for wl in b.workloads():
        w = b.g["workloads"][wl]
        if not w.get("gate_n"):
            continue
        cmd = [str(DIFF_PY), "-I", str(HERE / "dgen_gate.py"), "--root", str(g), "--workload", wl, "--kind",
               w["gate"], "--items", str(b.items(wl)), "--count", str(w["gate_n"]), "--tags", ",".join(tags),
               "--out", str(g / f"gate_{wl}.json")]
        env = {**os.environ, "HF_HOME": str(WORK / "hf"), "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": GATE_GPU}
        with open(g / f"gate_run_{wl}.log", "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    verdict = {}
    for wl in b.workloads():
        p = g / f"gate_{wl}.json"
        if not b.g["workloads"][wl].get("gate_n"):
            continue
        try:
            v = json.loads(p.read_text())["verdict"]
        except (OSError, ValueError, KeyError):
            failure(b.key, "-", "gate", wl, tail(g / f"gate_run_{wl}.log"), "gate produced no verdicts for this workload")
            continue
        for tag, x in v.items():
            prev = verdict.get(tag, {"verdict": "PASS", "detail": {}})
            prev["detail"][wl] = x
            if not str(x.get("verdict", "")).startswith("PASS"):
                prev["verdict"] = f"FAIL ({wl}: {x.get('verdict')})"
            verdict[tag] = prev
    (g / "gate.json").write_text(json.dumps({"verdict": verdict}, indent=1))
    b.decision("gate", verdicts={k: x["verdict"] for k, x in verdict.items()})
    for k, x in verdict.items():
        if not x["verdict"].startswith("PASS"):
            failure(b.key, k, "gate", "-", json.dumps(x)[:2000],
                    "knob set ineligible for selection" if "__" in k else
                    "gate FAIL recorded (sanity: clearly broken output); this arm's numbers are not reported as a win")


def select(b):
    """Rule: per competitor, the knob set with the best tune score (geometric-mean units/s over the tune
    points) among those whose sanity gate passed; else the fastest one, labelled gate-FAIL."""
    tune_res = json.loads((b.out / "tune.json").read_text()) if (b.out / "tune.json").exists() else {}
    try:
        verdict = json.loads((b.out / "gate" / "gate.json").read_text())["verdict"]
    except (OSError, ValueError, KeyError):
        verdict = {}
    tuned = b.tuned()
    bad = b.knob_failed()
    for arm, sets in tune_res.items():
        scored = sorted(((v["score"], k) for k, v in sets.items() if v.get("score") and f"{arm}__{k}" not in bad),
                        reverse=True)
        passing = [(s, k) for s, k in scored if verdict.get(f"{arm}__{k}", {}).get("verdict", "").startswith("PASS")]
        pick = (passing or scored or [(None, next(iter(b.knob_table(arm))))])[0]
        tuned[arm] = {"config": pick[1], "score": pick[0], "gate_pass": bool(passing), "candidates": scored}
        b.decision("tuned_config", engine=arm, config=pick[1], gate_pass=bool(passing), candidates=scored,
                   rule="best tune score among gate-passing knob sets")
    (b.out / "tuned.json").write_text(json.dumps(tuned, indent=1))


# ---------------------------------------------------------------- rounds / profile / summary
def rounds(b, n=3, first=1):
    arms = b.arms()
    for rnd in range(first, first + n):
        k = (rnd - 1) % len(arms)
        order = arms[k:] + arms[:k]
        log(f"== round {rnd}: {order}")
        for arm in order:
            b.gen_pass(arm, rnd, "main")


def means(b):
    latest = {}
    for r in b.rows():
        if r.get("tag") == "main" and "conc" in r:
            latest[(r["arm"], r["workload"], r["conc"], r["round"])] = r
    cells = {}
    for (arm, wl, c, _), r in latest.items():
        cells.setdefault((arm, wl, c), []).append(r)
    return cells


def profile(b):
    cells = means(b)
    mu = {k: statistics.fmean(r["units_per_s"] for r in v) for k, v in cells.items()
          if all(r.get("failed") == 0 and r.get("units_per_s") for r in v)}
    ratios = {}
    for (arm, wl, c), v in mu.items():
        if arm != "mstar-pr":
            continue
        comp = [x for (a, w, cc), x in mu.items() if w == wl and cc == c and a not in TREES]
        if comp:
            ratios[f"{wl}:{c}"] = v / max(comp)
    if ratios:
        pts = [min(ratios, key=ratios.get)]
        wl0 = pts[0].split(":")[0]
        if ratios.get(f"{wl0}:1", 2) < 1 and pts[0] != f"{wl0}:1":
            pts.append(f"{wl0}:1")
        rule = "lowest mstar-pr / best competitor units/s, plus c=1 if it trails there"
    else:
        wl0 = b.workloads()[0]
        pts = [f"{wl0}:{b.g['workloads'][wl0]['concs'][-1]}"]
        rule = "no competitor comparison: profile at the highest concurrency of the first workload"
    b.decision("profile_points", points=pts, ratios=ratios, rule=rule)
    t = TREES["mstar-pr"]
    for p in pts:
        wl, c = p.split(":")[0], int(p.split(":")[1])
        reap_all("before profile")
        lab = f"profile_{wl}_c{c}"
        _, kargs = b.knobs_of("mstar-pr")
        name = b.start("mstar-pr", lab, 0, PORT0, kargs, env_extra={"MSTAR_PHASE_TIMING": "100"})
        if not name:
            failure(b.key, "mstar-pr", "profile", p, tail(b.out / "servers" / f"{lab}.log"), "no profile")
            continue
        try:
            w = b.g["workloads"][wl]
            b.client("mstar-pr", wl, b.out / "profile" / f"client_{wl}_c{c}", c, w["nums"][str(c)], 2)
        finally:
            b.stop(name)
            reap_all("after profile")
        code = ("import sys; from benchmark.worker_phases.phases import parse_log, segment, render; "
                "print(render(segment(parse_log(open(sys.argv[1]).read()))))")
        r = subprocess.run([f"{t}/.venv/bin/python", "-c", code, str(b.out / "servers" / f"{lab}.log")], cwd=t,
                           capture_output=True, text=True, timeout=600)
        (b.out / "profile").mkdir(exist_ok=True)
        (b.out / "profile" / f"phases_{wl}_c{c}.txt").write_text(r.stdout + r.stderr)


def summarize(b):
    cells = means(b)
    arms = b.arms()
    u = b.g.get("units", "units")
    L = [f"# {b.key}: {b.g.get('title', b.key)}", "",
         b.g.get("summary_note", ""), "",
         "Arms: " + "; ".join(f"{b.label(a)} = {b.knobs_of(a)[0]}" for a in arms),
         "Competitor choice: " + b.g.get("competitor_note", "-"), "",
         f"Each cell: mean [min-max] over rounds (n), each round a fresh server process; FAILED if any round "
         f"failed. Offline-batch arms: the official package's own batch inference, batch b in the c column (no "
         f"server, queueing or HTTP cost).", ""]
    if not b.comps():
        L += ["**No competitor could be run for this model: mstar-pr vs mstar-main only.**", ""]
    for wl in b.workloads():
        w = b.g["workloads"][wl]
        concs = sorted({c for (a, x, c) in cells if x == wl})
        if not concs:
            continue
        L += [f"## {wl}: {w.get('desc', '')}", ""]
        for metric, unit, fd in (("units_per_s", f"{u}/s", 3), ("throughput_per_s", "req/s", 3),
                                 ("lat_p50", "latency p50 (s)", 3), ("lat_p99", "latency p99 (s)", 3),
                                 ("ttfb_p50", "TTFB p50 (s)", 3)):
            L += [f"### {unit}", "", "| arm | " + " | ".join(f"c={c}" for c in concs) + " |",
                  "|---|" + "---:|" * len(concs)]
            for a in arms:
                cols = []
                for c in concs:
                    rs = cells.get((a, wl, c), [])
                    if not rs:
                        cols.append("-")
                    elif any(r.get("failed") != 0 for r in rs):
                        cols.append(f"FAILED (n={len(rs)})")
                    else:
                        v = [r[metric] for r in rs if r.get(metric) is not None]
                        cols.append(f"{statistics.fmean(v):.{fd}f} [{min(v):.{fd}f}-{max(v):.{fd}f}] n={len(v)}"
                                    if v else "-")
                L.append(f"| {b.label(a)} | " + " | ".join(cols) + " |")
            L.append("")
        pr = {c: statistics.fmean(r["units_per_s"] for r in cells[("mstar-pr", wl, c)])
              for c in concs if ("mstar-pr", wl, c) in cells and all(
                  r.get("failed") == 0 and r.get("units_per_s") for r in cells[("mstar-pr", wl, c)])}
        rel = []
        for a in arms:
            if a == "mstar-pr":
                continue
            xs = []
            for c in concs:
                rs = cells.get((a, wl, c), [])
                if c in pr and rs and all(r.get("failed") == 0 and r.get("units_per_s") for r in rs):
                    xs.append(f"c={c} {100 * (pr[c] / statistics.fmean(r['units_per_s'] for r in rs) - 1):+.1f}%")
            if xs:
                rel.append(f"- M* PR vs {b.label(a)}: " + ", ".join(xs))
        if rel:
            L += [f"M* PR {u}/s relative to each arm:", ""] + rel + [""]
    g = b.out / "gate" / "gate.json"
    if g.exists():
        v = json.loads(g.read_text())["verdict"]
        L += ["## Sanity gate (dgen_gate.py; parity annotated, not gating)", "", "| tag | verdict | detail |",
              "|---|---|---|"]
        for k, x in v.items():
            L.append(f"| {k} | {x['verdict']} | " + "; ".join(
                f"{wl}: {json.dumps({kk: vv for kk, vv in d.items() if kk != 'verdict'})[:300]}"
                for wl, d in x["detail"].items()) + " |")
        L.append("")
    for p in sorted((b.out / "profile").glob("phases_*.txt")) if (b.out / "profile").exists() else []:
        L += [f"Profile: {p.name} (MSTAR_PHASE_TIMING=100, benchmark/worker_phases)", ""]
    (b.out / "summary.md").write_text("\n".join(L) + "\n")
    notes = [f"notes for {b.key}", "dropped engines: " + json.dumps(b.dropped())]
    p = b.out / "startup.jsonl"
    if p.exists():
        st = {}
        for x in p.read_text().splitlines():
            r = json.loads(x)
            st.setdefault(r["arm"], []).append(r["seconds"])
        notes.append("startup s: " + "; ".join(f"{a}: median {statistics.median(v):.0f} n={len(v)}"
                                               for a, v in st.items()))
    for log_ in sorted((b.out / "servers").glob("main_mstar*.log")):
        txt = log_.read_text(errors="replace")
        notes.append(f"{log_.stem}: failed captures {txt.count('ailed to capture')}, graph runtime rust "
                     f"{'graph runtime: rust' in txt}")
    (b.out / "notes.txt").write_text("\n".join(notes) + "\n")


def run_phase(q, spec, phase):
    b = GenBench(spec)
    if phase == "setup":
        setup(b)
    elif phase == "tune":
        tune(b)
    elif phase == "collect":
        collect(b)
    elif phase == "gate":
        gate(b)
    elif phase == "select":
        select(b)
    elif phase == "rounds":
        rounds(b, q.get("rounds", 3))
    elif phase == "rounds_extra":
        rounds(b, 2, first=4)
    elif phase == "profile":
        profile(b)
    elif phase in ("summarize", "resummarize"):
        summarize(b)
    else:
        raise ValueError(phase)
