"""Resumable queue of cells, run one at a time on GPU 0 (timed work never overlaps).

  phases.py <phase> [<phase> ...]     run the named phases' cells in order; a cell with DONE is skipped
  phases.py --list

Each cell is one q3.py cell (fresh server -> slo_bench -> stop). Results under /workspace/q3/results/<sub>.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get("Q3_BASE", "/workspace"))
W = BASE / "q3"
RES = W / "results"
PY = str(W / "main/.venv/bin/python")
Q3 = str(W / "harness/q3.py")
T = str(W / "harness")

RATES_A = "2,4,6,8,10,12,14,16,18,20,24,28,32"
RATES_B = "4,8,12,16,20,24,32,40,48,56,64,72,80,96"
SWEEP_A = ["--texts", f"{T}/corpus_a.jsonl", "--rates", RATES_A, "--timeout", "180"]
SWEEP_B = ["--texts", f"{T}/seedtts_en.jsonl", "--shuffle", "--rates", RATES_B, "--timeout", "120"]
CORR = ["--texts", f"{T}/corr.jsonl", "--concurrency", "8"]
PROF = ["--texts", f"{T}/seedtts_en.jsonl", "--requests", "512"]
TIMING = ["MSTAR_PHASE_TIMING=100"]
VOX_ARGS = ["--max-tokens", "2048"]
VOX_SAFE = ["--max-num-pages", "1024"]
VOX_ENV = ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]


def cell(sub, arm, kind, bench, env=(), serve=(), tag=None):
    return {"sub": sub, "arm": arm, "kind": kind, "bench": list(bench), "env": list(env), "serve": list(serve),
            "tag": tag or sub.replace("/", "_")}


def vox(extra):
    return [*VOX_ARGS, *extra]


PHASES = {
    # corr_m runs on the mstar queue, corr_v on the competitor queue (GPU-parallel); "corr" is both, in order
    "corr_m": [
        cell("corr/main_a", "main", "collect", CORR),
        cell("corr/stack_b", "stack", "collect", CORR),
    ],
    "corr_v": [
        cell("corr/voxserve", "voxserve", "collect", CORR, serve=vox(["--max-batch-size", "32"])),
    ],
    "sgo_tune": [  # SGLang-Omni runs once: pick the better of default / 10-token chunks on a short B probe
        cell("tune/sgo_default", "sgomni", "sweep", [*SWEEP_B[:3], "--shuffle", "--rates", "16,32,48", "--window", "30"]),
        cell("tune/sgo_10tok", "sgomni", "sweep", [*SWEEP_B[:3], "--shuffle", "--rates", "16,32,48", "--window", "30"],
             env=[f"Q3_SGO_CONFIG={T}/sgomni_qwen3tts_1p7b_10tok.yaml"]),
    ],
    "base_r1": [
        cell("slo/A/stack_r1", "stack", "sweep", SWEEP_A),
        cell("slo/A/main_r1", "main", "sweep", SWEEP_A),
        cell("slo/B/main_r1", "main", "sweep", SWEEP_B),
        cell("slo/B/stack_r1", "stack", "sweep", SWEEP_B),
    ],
    "prof_base": [
        cell("prof/stack_c32", "stack", "closed", [*PROF, "--concurrency", "32"], env=TIMING),
        cell("prof/stack_c1", "stack", "closed", [*PROF, "--concurrency", "1", "--requests", "48"], env=TIMING),
        cell("prof/main_c32", "main", "closed", [*PROF, "--concurrency", "32"], env=TIMING),
        cell("prof/stack_c64", "stack", "closed", [*PROF, "--concurrency", "64"], env=TIMING),
    ],
    # VoxServe ce212de19 holds ~133 GB at b32 on a 141 GB H200 and OOMs under overload at its defaults; b64 OOMs
    # in graph capture. Every tune cell runs with 1024 KV pages (enough for 64 x 2048-token requests at 128/page
    # here) and expandable segments, which only add headroom. online = its deadline-aware ("pressing") scheduler.
    "vox_tune": [
        cell(f"tune/vox_{name}", "voxserve", "sweep", [*SWEEP_B[:3], "--shuffle", "--rates", "16,32,48", "--window", "30"],
             env=VOX_ENV, serve=vox([*VOX_SAFE, *flags]))
        for name, flags in [
            ("b32_base", ["--max-batch-size", "32"]),
            ("b32_online_async", ["--max-batch-size", "32", "--scheduler-type", "online", "--async-scheduling"]),
            ("b32_online_async_unroll", ["--max-batch-size", "32", "--scheduler-type", "online", "--async-scheduling",
                                         "--unroll-depth-cuda-graph"]),
            ("b32_online_async_unroll_compile", ["--max-batch-size", "32", "--scheduler-type", "online", "--async-scheduling",
                                                 "--unroll-depth-cuda-graph", "--enable-torch-compile"]),
            ("b16_online_async_unroll", ["--max-batch-size", "16", "--scheduler-type", "online", "--async-scheduling",
                                         "--unroll-depth-cuda-graph"]),
        ]
    ],
}


def vox_best():
    """Serve args of the best vox_tune cell (or tune/vox_best.json when written by hand)."""
    p = RES / "tune/vox_best.json"
    if p.exists():
        return json.loads(p.read_text())["serve"]
    cands = [(c["sub"], c["serve"]) for c in PHASES["vox_tune"] if (RES / c["sub"] / "DONE").exists()]
    return best_of(cands)[1]


def best_of(cands):
    """The tune cell (sub, env) with the highest sum of attainment over its probe rates."""
    def score(sub):
        p = RES / sub / "summary.jsonl"
        return sum(json.loads(x)["attainment"] or 0 for x in open(p)) if p.exists() else -1
    return max(cands, key=lambda c: score(c[0]))


def sgo_env():
    return best_of([("tune/sgo_default", []), ("tune/sgo_10tok", [f"Q3_SGO_CONFIG={T}/sgomni_qwen3tts_1p7b_10tok.yaml"])])[1]


ORPH = ["Q3_MODEL=orpheus", f"Q3_CONFIG={T}/orpheus_unsloth.yaml"]
CHAT = ["Q3_MODEL=chatterbox", "Q3_CONFIG=configs/chatterbox.yaml"]
# Orpheus/Chatterbox: Seed-TTS B sentences, closed-loop throughput is the shared metric with VoxServe
TTS_B = ["--texts", f"{T}/seedtts_en.jsonl", "--shuffle", "--timeout", "120"]


def kind_cells(kind, env, vox_model):
    rates = "2,4,8,12,16,24,32,48,64" if kind == "orpheus" else "2,4,6,8,12,16,24,32"
    voice = "tara" if kind == "orpheus" else "default"
    bench = [*TTS_B, "--rates", rates, "--kind", kind, "--voice", voice]
    vox_env = [f"Q3_VOX_MODEL={vox_model}"]
    return [cell(f"slo/{kind}/main_r1", "main", "sweep", bench, env=env),
            cell(f"slo/{kind}/stack_r1", "stack", "sweep", bench, env=env),
            cell(f"slo/{kind}/vox_r1", "voxserve", "sweep", bench, env=[*vox_env, *VOX_ENV], serve=vox(["--max-batch-size", "32"])),
            cell(f"prof/{kind}_stack_c32", "stack", "closed",
                 ["--texts", f"{T}/seedtts_en.jsonl", "--requests", "256", "--concurrency", "32", "--kind", kind,
                  "--voice", voice], env=[*env, *TIMING])]


SPLIT_MPS = [f"Q3_CONFIG={T}/qwen3tts_1p7b_split_mps.yaml"]


def dynamic(name):
    if name == "corr":
        return [*PHASES["corr_m"], *PHASES["corr_v"]]
    if name == "split_mps":  # stack head, Codec on its own worker on the same GPU, both MPS clients
        return [cell("slo/B/stack_split_mps_r1", "stack", "sweep", SWEEP_B, env=SPLIT_MPS),
                cell("slo/A/stack_split_mps_r1", "stack", "sweep", SWEEP_A, env=SPLIT_MPS),
                cell("prof/stack_split_mps_c32", "stack", "closed", [*PROF, "--concurrency", "32"], env=[*SPLIT_MPS, *TIMING])]
    if name == "comp_sgo_r1":
        e = sgo_env()
        return [cell("slo/B/sgomni_r1", "sgomni", "sweep", SWEEP_B, env=e),
                cell("slo/A/sgomni_r1", "sgomni", "sweep", SWEEP_A, env=e),
                cell("corr/sgomni", "sgomni", "collect", CORR, env=e)]
    for kind, env, vm in (("orpheus", ORPH, "unsloth/orpheus-3b-0.1-ft"), ("chatterbox", CHAT, "ResembleAI/chatterbox")):
        cells = kind_cells(kind, env, vm)  # _m: mstar arms (mstar queue), _v: VoxServe (competitor queue)
        if name == kind:
            return cells
        if name == f"{kind}_m":
            return [c for c in cells if c["arm"] != "voxserve"]
        if name == f"{kind}_v":
            return [c for c in cells if c["arm"] == "voxserve"]
    if name == "comp_vox_r1":
        return [cell("slo/B/vox_r1", "voxserve", "sweep", SWEEP_B, env=VOX_ENV, serve=vox_best()),
                cell("slo/A/vox_r1", "voxserve", "sweep", SWEEP_A, env=VOX_ENV, serve=vox_best()),
                cell("corr/vox_best", "voxserve", "collect", CORR, env=VOX_ENV, serve=vox_best())]
    if name == "base_r2":  # round 2: fresh servers, order rotated
        return [cell("slo/B/stack_r2", "stack", "sweep", SWEEP_B), cell("slo/A/main_r2", "main", "sweep", SWEEP_A),
                cell("slo/B/main_r2", "main", "sweep", SWEEP_B), cell("slo/A/stack_r2", "stack", "sweep", SWEEP_A)]
    if name.startswith("fix:"):  # fix:<arm>:<label> -- a READY head checked out as tree <arm>
        _, arm, label = name.split(":")
        return [cell(f"slo/A/{label}_r1", arm, "sweep", SWEEP_A), cell(f"slo/B/{label}_r1", arm, "sweep", SWEEP_B),
                cell(f"prof/{label}_c32", arm, "closed", [*PROF, "--concurrency", "32"], env=TIMING),
                cell(f"corr/{label}", arm, "collect", CORR)]
    return PHASES[name]


GPU = 0  # phases.py --gpu N: this queue's GPU (port 8100+N); queues on different GPUs run side by side


def run(c):
    out = RES / c["sub"]
    if (out / "DONE").exists():
        print(f"skip {c['sub']} (DONE)", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, Q3, "cell", c["arm"], c["kind"], c["sub"], "--tag", c["tag"], "--gpu", str(GPU), "--port", str(8100 + GPU)]
    for e in c["env"]:
        cmd += ["--env", e]
    for s in c["serve"]:
        cmd += [f"--serve-arg={s}"]
    cmd += ["--", *c["bench"]]
    t0 = time.time()
    # host CPU matters in SLO sweeps: anyone borrowing spare GPUs on this box pauses while a PAUSE file exists
    pause = BASE / f"codec/PAUSE.gpu{GPU}"
    if c["kind"] == "sweep":
        pause.parent.mkdir(parents=True, exist_ok=True)
        pause.write_text(f"{c['sub']} since {time.strftime('%F %T')}\n")
    print(f"== {time.strftime('%T')} cell {c['sub']}: {' '.join(cmd)}", flush=True)
    try:
        with open(out / "cell.out", "a") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    finally:
        pause.unlink(missing_ok=True)
    print(f"   rc={rc} {time.time() - t0:.0f}s; tail:\n" + "".join(open(out / "cell.out").readlines()[-4:]), flush=True)
    if rc == 0 and (W / c["arm"] / ".venv/bin/mstar").exists() and any(e.startswith("MSTAR_PHASE_TIMING") for e in c["env"]):
        tree = W / c["arm"]  # render the phase table with the arm's own benchmark.worker_phases
        subprocess.run([str(tree / ".venv/bin/python"), str(W / "harness/prof_render.py"), str(RES / "servers" / f"{c['tag']}.log"),
                        str(out / "phases.md")], cwd=tree, stdout=open(out / "render.out", "w"), stderr=subprocess.STDOUT)
    if rc == 0:
        (out / "DONE").write_text(time.strftime("%F %T"))
    with open(RES / "phase_marks.txt", "a") as fh:
        fh.write(f"{c['sub']} gpu{GPU} rc={rc} {int(time.time())}\n")


def main():
    global GPU
    if len(sys.argv) > 2 and sys.argv[1] == "--gpu":
        GPU = int(sys.argv[2])
        del sys.argv[1:3]
    if sys.argv[1:] == ["--list"]:
        for k, v in PHASES.items():
            print(k, [c["sub"] for c in v])
        return
    for name in sys.argv[1:]:
        for c in dynamic(name):
            run(c)
        print(f"PHASE_DONE {name} {time.strftime('%T')}", flush=True)


if __name__ == "__main__":
    main()
