#!/usr/bin/env python3
"""Unattended queue runner for node2 (/root/work/dbench/dqueue.py, self-contained in /root/work/dbench).

Re-reads queue.json before every phase, and reloads the phase modules (dbench, dtts), so entries can be
appended and harness fixes picked up without restarting it. Entries run in order, one at a time; each
entry is a list of phases; every phase is idempotent and resumable (stage markers under
<results>/<entry>/stages/q_<phase>.done, timed cells with failed == 0 are skipped). supervisor.py reruns
this file if it dies. The heartbeat (RESULTS/heartbeat.txt, every 30 s) gives the queue position.

Diffusion phases (dbench): setup, knobs, inherit, collect, tune, ref_c, gate, select, gallery, rounds,
  rounds_extra, profile, summarize.  TTS phases: see dtts.py. Entries with "module": "<name>" (e.g. dgen:
  ASR, BAGEL, world/video models) run that module's run_phase(q, spec, phase).
Touch /root/work/dbench/.reload to make the runner exit at the next phase boundary (planned restart).

Decision rules (applied here, logged to RESULTS/decisions.jsonl):
  - tuned competitor config = best tune score among knob sets whose gate verdict passed (dbench.phase_select)
  - gate verdicts = the fixed criteria in dgate.py / dtts.py
  - profile point = where mstar-pr / best competitor throughput is lowest (+ c=1 if it trails there)
  - a phase that crashes is retried by the supervisor; after 3 crashes it is abandoned and logged
  - an optional "deadline_utc" in queue.json stops new timed work after that time (unset = none)
"""
from __future__ import annotations

import calendar
import importlib
import json
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dbench  # noqa: E402

RESULTS = dbench.RESULTS
HB = RESULTS / "heartbeat.txt"
POS = {"text": "starting"}


def load_queue():
    q = json.loads((HERE / "queue.json").read_text())
    dl = q.get("deadline_utc")
    dbench.DEADLINE = calendar.timegm(time.strptime(dl, "%Y-%m-%dT%H:%M:%SZ")) if dl else float("inf")
    return q


def heartbeat_loop():
    while True:
        s = dbench.STATUS
        try:
            HB.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} | {POS['text']} | "
                          f"model={s['model']} | phase={s['phase']} | arm={s['arm']} | cell={s['cell']}\n")
        except OSError:
            pass
        time.sleep(30)


def entry_bench(spec):
    """A diffusion entry's Bench; variants ("parent": key, "overrides": {...}) get their own MODELS row."""
    key = spec["key"]
    if key not in dbench.MODELS:
        base = dict(dbench.MODELS[spec["parent"]])
        base.update(spec.get("overrides", {}))
        base["nums"] = {int(k): v for k, v in base["nums"].items()}
        base["concs"] = tuple(base["concs"])
        base["dref_model"] = dbench.MODELS[spec["parent"]].get("dref_model", spec["parent"])
        dbench.MODELS[key] = base
    return dbench.Bench(key)


def run_diffusion_phase(q, spec, phase):
    b = entry_bench(spec)
    m = b.m
    arms = spec["arms"]
    served_mstar = [a for a in arms if a.startswith("mstar")]
    comps = [a for a in arms if a in ("vllmomni", "sglang")]
    variant = bool(spec.get("parent"))
    if phase == "setup":
        dbench.setup_diffusion(b)
    elif phase == "knobs":
        (b.out / "knobs.json").write_text(json.dumps(q["knobs"].get(b.key) or q["knobs"][m["kind"]], indent=1))
    elif phase == "inherit":  # a variant reuses its parent's knob table and tuned configs
        parent = RESULTS / spec["parent"]
        for f in ("knobs.json", "tuned.json"):
            (b.out / f).write_text((parent / f).read_text())
        b.decision("inherit", parent=spec["parent"], tuned=json.loads((b.out / "tuned.json").read_text()))
    elif phase == "collect":  # a variant also collects its competitors' tuned configs here
        b.phase_collect(served_mstar + (comps if variant else []), ["ref_a", "ref_b", "ref_k", "ref_seed"],
                        list(range(8)))
    elif phase == "tune":
        b.phase_tune(comps + (["diffusers"] if "diffusers" in arms else []))
    elif phase == "ref_c":
        b.phase_collect([], ["ref_c"], list(range(8)))
    elif phase == "gate":
        b.phase_gate(served_mstar + (comps if variant else []))
    elif phase == "select":
        b.phase_select()
    elif phase == "gallery":
        tuned = b.tuned()
        ctags = comps if variant else [f"{e}__{tuned[e]['config']}" for e in comps if e in tuned]
        tags = ["ref_a", "ref_c"] + [a.replace("-", "_") for a in served_mstar] + ctags
        items = "0,8,16,24" if m["kind"] == "image" else "0,1,2,3"
        subprocess.run([str(dbench.DIFF_PY), "-I", str(HERE / "dgallery.py"), "--root", str(b.out / "gate"), "--kind",
                        m["kind"], "--tags", ",".join(tags), "--items", items, "--out",
                        str(b.out / f"sheet_{b.key}.jpg")], timeout=900)
    elif phase == "rounds":
        b.phase_rounds(arms, spec.get("rounds", q.get("rounds", 3)))
    elif phase == "rounds_extra":  # backlog: rounds 4-5
        b.phase_rounds(arms, 2, first=4)
    elif phase == "profile":
        for conc in pick_profile_points(b, arms):
            b.phase_profile(conc, m["nums"][conc])
    elif phase in ("summarize", "resummarize"):
        summarize(b)
    else:
        raise ValueError(phase)


def means(b, tag="main"):
    latest = {}
    for r in b.rows():
        if r.get("tag") == tag and "conc" in r and r.get("steps") == b.m["steps"]:
            latest[(r["arm"], r["conc"], r["round"])] = r
    out = {}
    for (arm, c, _), r in latest.items():
        if r.get("failed") == 0 and r.get("throughput_per_s"):
            out.setdefault((arm, c), []).append(r["throughput_per_s"])
    return {k: statistics.fmean(v) for k, v in out.items()}


def pick_profile_points(b, arms):
    mu = means(b)
    ratios = {}
    for c in b.m["concs"]:
        comp = [mu[(a, c)] for a in arms if not a.startswith("mstar") and (a, c) in mu]
        if ("mstar-pr", c) in mu and comp:
            ratios[c] = mu[("mstar-pr", c)] / max(comp)
    if not ratios:
        pts, rule = [b.m["concs"][-1]], "no complete comparison; profile at the highest concurrency"
    else:
        worst = min(ratios, key=ratios.get)
        pts = [worst] + ([1] if worst != 1 and ratios.get(1, 2) < 1 else [])
        rule = "point where mstar-pr / best competitor is lowest, plus c=1 if mstar-pr trails there"
    b.decision("profile_points", points=pts, ratios=ratios, rule=rule)
    return pts


def health(b):
    import re
    out = {}
    for log in sorted((b.out / "servers").glob("*.log")):
        txt = log.read_text(errors="replace")
        h = {}
        if "mstar" in log.name:
            h["failed_capture"] = len(re.findall(r"[Ff]ailed to capture", txt))
            h["graph_runtime_rust"] = "graph runtime: rust" in txt
            h["recompile_lines"] = len(re.findall(r"[Rr]ecompil", txt))
        elif "vllmomni" in log.name:
            h["attention"] = sorted(set(re.findall(r"Resolved diffusion attention backend '(\w+)'", txt)))
            h["regional_compile"] = "Regional compilation applied" in txt
            h["enforce_eager"] = "enforce_eager=True" in txt
        elif "sglang" in log.name:
            h["attention"] = sorted(set(re.findall(r"Attention backends for transformer: (\w+)", txt)))
            h["speed_mode_msgs"] = sorted(set(re.findall(r"performance_mode=speed [^\n]{0,60}", txt)))[:3]
            h["offload_msgs"] = sorted(set(re.findall(r"[^\n]{0,40}layerwise offload[^\n]{0,40}", txt)))[:3]
        h["errors"] = len(re.findall(r"Traceback|CUDA out of memory|OutOfMemoryError", txt))
        out[log.stem] = h
    (b.out / "health.json").write_text(json.dumps(out, indent=1))
    return out


def summarize(b):
    subprocess.run([sys.executable, "-I", str(HERE / "dsummary.py"), "--out", str(b.out), "--kind", b.m["kind"]],
                   timeout=600, capture_output=True)
    h = health(b)
    st = {}
    p = b.out / "startup.jsonl"
    if p.exists():
        for x in p.read_text().splitlines():
            r = json.loads(x)
            st.setdefault(r["arm"], []).append(r["seconds"])
    dec = [d for d in (json.loads(x) for x in (RESULTS / "decisions.jsonl").read_text().splitlines())
           if d.get("model") == b.key] if (RESULTS / "decisions.jsonl").exists() else []
    lines = [f"notes for {b.key} ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})", "",
             "server startup seconds per arm (all starts, incl. cold compiles): " +
             "; ".join(f"{a}: min {min(v):.0f} median {statistics.median(v):.0f} max {max(v):.0f} n={len(v)}"
                       for a, v in st.items()), "", "health (per server log):"]
    lines += [f"  {k}: {json.dumps(v)}" for k, v in h.items() if k.startswith(("main_", "tune_"))]
    lines += ["", "decisions:"] + [
        f"  {d['time']} {d['kind']}: "
        f"{json.dumps({k: v for k, v in d.items() if k not in ('time', 'model', 'kind')})[:300]}" for d in dec]
    (b.out / "notes.txt").write_text("\n".join(lines) + "\n")
    subprocess.run([sys.executable, "-I", str(HERE / "report_diffusion.py")], timeout=600)


def run_phase(q, spec, phase):
    global dbench
    dbench = importlib.reload(dbench)
    dbench.STATUS.update(model=spec["key"], phase=phase)
    load_queue()  # re-applies the deadline to the reloaded module
    # an entry may name its phase module ("module": "dgen"); TTS entries use dtts. Reloaded per phase.
    mod_name = spec.get("module") or ("dtts" if spec.get("kind") == "tts" else None)
    if mod_name:
        mod = importlib.reload(importlib.import_module(mod_name))
        mod.run_phase(q, spec, phase)
    else:
        run_diffusion_phase(q, spec, phase)


def skip_entry(spec):
    stages = RESULTS / spec["key"] / "stages"
    if (stages / "q_skipped.done").exists():
        return True
    if spec.get("min_hours_left") and dbench.DEADLINE != float("inf") and \
            not (stages / "q_setup.done").exists() and \
            (dbench.DEADLINE - time.time()) / 3600 < spec["min_hours_left"]:
        stages.mkdir(parents=True, exist_ok=True)
        (stages / "q_skipped.done").touch()
        with (RESULTS / "decisions.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": spec["key"],
                                "kind": "model_skipped", "min_hours_left": spec["min_hours_left"]}) + "\n")
        return True
    return False


def next_work(q):
    for i, spec in enumerate(q["models"]):
        if skip_entry(spec):
            continue
        stages = RESULTS / spec["key"] / "stages"
        for j, phase in enumerate(spec["phases"]):
            if not (stages / f"q_{phase}.done").exists():
                return i, spec, j, phase
    return None


def main():
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    while True:
        q = load_queue()
        nxt = next_work(q)
        if nxt is None:  # queue empty: say DONE, keep polling queue.json for appended entries
            if not (RESULTS / "DONE").exists():
                (RESULTS / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC\n", time.gmtime()))
                dbench.log("==== queue empty: DONE (polling queue.json for new entries)")
            POS["text"] = "DONE (queue empty, polling queue.json)"
            dbench.STATUS.update(model="ALL", phase="DONE", arm=None, cell=None)
            time.sleep(60)
            continue
        (RESULTS / "DONE").unlink(missing_ok=True)
        i, spec, j, phase = nxt
        if (HERE / ".reload").exists():  # touch it: restart at this phase boundary (no failure logged)
            (HERE / ".reload").unlink()
            (HERE / ".planned_restart").touch()
            dbench.log("==== reload requested: exiting at a phase boundary for the supervisor to restart")
            with (RESULTS / "planned_restarts.jsonl").open("a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                                    "note": f"reload flag: restart before {spec['key']} {phase}"}) + "\n")
            sys.exit(0)
        key = spec["key"]
        stages = RESULTS / key / "stages"
        stages.mkdir(parents=True, exist_ok=True)
        POS["text"] = f"queue {i + 1}/{len(q['models'])} {key} phase {j + 1}/{len(spec['phases'])} {phase}"
        dbench.STATUS.update(model=key, phase=phase, arm=None, cell=None)
        dbench.log(f"==== {POS['text']}")
        try:
            dbench.reap_all(f"before {key} {phase}")
            run_phase(q, spec, phase)
            dbench.reap_all(f"after {key} {phase}")
            (stages / f"q_{phase}.done").touch()
        except Exception as e:  # noqa: BLE001
            dbench.log(f"!! {key} {phase}: {e!r}\n{traceback.format_exc()}")
            crashes = stages / f"q_{phase}.crashes"
            n = int(crashes.read_text()) + 1 if crashes.exists() else 1
            crashes.write_text(str(n))
            dbench.reap_all(f"after failed {phase}")
            with (RESULTS / "decisions.jsonl").open("a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": key,
                                    "kind": "phase_error", "phase": phase, "crashes": n,
                                    "error": repr(e)[:300]}) + "\n")
            dbench.failure(key, "-", phase, dbench.STATUS.get("cell"), traceback.format_exc()[-3000:],
                           "runner exits; supervisor restarts it and the phase resumes" if n < 3 else
                           "third crash of this phase: entry marked failed (skipped); queue moves to the next entry")
            if n < 3:
                raise  # supervisor restarts us; finished cells are skipped
            # third crash of the same phase: the entry is marked failed (skipped) and the queue moves on
            (stages / "q_skipped.done").touch()


if __name__ == "__main__":
    main()
