"""Same-box baselines for push4-newb's generation/action models (push4c GPUs 0-3, even CPUs), driven by
dgen_client.py (bench1021's spec-driven closed-loop client: one client and policy for every engine).

  run_gen.py items  --model droid                       (writes ~/push4/newb/data/droid.jsonl)
  run_gen.py bench  --model droid --arms base,sglang_b1,sglang_b8,base2 --rounds 2 --out DIR
  run_gen.py profile --model droid --out DIR            (mstar with MSTAR_PHASE_TIMING=100, untimed)

Arms: "base"/"base2"/<name>[:K=V;...] = mstar from ~/push4/newb/mstar (env overrides; OVN_CONFIG=<path>);
competitor arms are named in MODELS[m]["comps"]. Each arm gets a fresh server per round, rounds rotate the order.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
NEWB = HOME / "push4" / "newb"
TREE = NEWB / "mstar"
PY = str(NEWB / "venv" / "bin" / "python")
ENG = NEWB / "engines"
DATA = NEWB / "data"
HERE = Path(__file__).resolve().parent
CORES = os.environ.get("PUSH4_CORES", ",".join(str(c) for c in range(0, 207, 2)))
# mstar servers can get their own small CPU set: on push4c's schedutil host, mstar's blocking thread hand-offs
# run on cores that never clock up when spread wide (q35small). Competitors and clients keep CORES.
MSTAR_CORES = os.environ.get("PUSH4_MSTAR_CORES", CORES)
PORT = int(os.environ.get("PUSH4_PORT", "59200"))  # SGLang also takes PORT + 1

DROID_HF = "nvidia/Cosmos3-Nano-Policy-DROID"
SUPER_I2V_HF = "nvidia/Cosmos3-Super-Image2Video-4Step"
DROID_MK = {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10, "action_chunk_size": 32,
            "num_frames": 33, "seed": 0, "num_inference_steps": 4, "guidance_scale": 3.0, "flow_shift": 5.0,
            "size": "832x480", "fps": 15}
MODELS = {
    "droid": dict(
        mstar="cosmos3_droid", config="configs/cosmos3_droid.yaml", gpus="0", units="chunks",
        concs=[1, 4, 8], nums={1: 24, 4: 48, 8: 96}, warm_conc=8, warm_n=32,
        spec={
            "mstar": {"path": "/generate", "encoding": "multipart",
                      "body": {"text": "{task}", "input_modalities": "image,text", "output_modalities": "action",
                               "streaming": "false", "model_kwargs": "{model_kwargs}"},
                      "files": {"files": "{image_path}"}, "response": "json", "extract": "outputs.action.0.data",
                      "decode": "b64", "check": "bytes", "save": "bin"},
            "sglang": {"path": "/v1/actions/generations", "encoding": "json",
                       "body": {"model": DROID_HF,
                                "input": {"task": "{task}", "observation": {"image": {"b64_json": "{image_b64}"}}},
                                "parameters": {"action_mode": "policy", "domain_name": "droid_lerobot",
                                               "raw_action_dim": 10, "action_horizon": 32, "num_frames": 33,
                                               "width": 832, "height": 480, "fps": 15, "num_inference_steps": 4,
                                               "guidance_scale": 3.0, "flow_shift": 5.0, "seed": "{seed}"}},
                       "response": "json", "extract": "data.0.action.values", "check": "json", "save": "json"}},
        comps={"sglang_b1": ("sglang", []), "sglang_b8": ("sglang", ["--batching-max-size", "8"])},
        sglang_model=DROID_HF,
    ),
    # Cosmos3-Super image-to-video, 4-step distilled, TP2: the model card's recipe (480p, 189 frames, fps 24,
    # guidance 1.0; steps and sigmas are fixed by the checkpoint) on its own example frame and JSON prompt.
    "super_i2v": dict(
        mstar="cosmos3_super_i2v_4step", config="configs/cosmos3_super_i2v_4step_tp2.yaml", gpus="0,1",
        units="frames", concs=[1, 2], nums={1: 6, 2: 8}, warm_conc=2, warm_n=4,
        spec={
            "mstar": {"path": "/v1/videos/generations", "encoding": "json",
                      "body": {"prompt": "{prompt}", "image": "{image_data_uri}", "size": "832x480",
                               "num_frames": 189, "fps": 24, "guidance_scale": 1.0, "seed": "{seed}"},
                      "response": "json", "extract": "data.0.b64_json", "decode": "b64", "check": "video",
                      "save": "mp4"},
            "sglang": {"path": "/v1/videos", "encoding": "multipart",
                       "body": {"model": SUPER_I2V_HF, "prompt": "{prompt}", "size": "832x480", "num_frames": "189",
                                "fps": "24", "guidance_scale": "1.0", "seed": "{seed}"},
                       "files": {"input_reference": "{image_path}"}, "response": "json",
                       "poll": {"status_path": "/v1/videos/{id}", "content_path": "/v1/videos/{id}/content",
                                "id_key": "id", "done": "completed"},
                       "check": "video", "save": "mp4"}},
        comps={"sglang": ("sglang", ["--tp-size", "2"])},
        sglang_model=SUPER_I2V_HF,
    ),
}


def log(msg: str) -> None:
    print(f"[gen {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pin(cmd: list[str], cores: str = CORES) -> list[str]:
    return ["numactl", f"--physcpubind={cores}"] + cmd


def healthy(port: int) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
    return False


def parse_arm(spec: str) -> tuple[str, dict]:
    name, _, envs = spec.partition(":")
    return name, (dict(kv.split("=", 1) for kv in envs.split(";") if kv) if envs else {})


class Server:
    def __init__(self, m: dict, arm: str, env_extra: dict, port: int, logpath: Path):
        self.port = port
        if healthy(port):
            raise RuntimeError(f"port {port} already serving")
        env = dict(os.environ)
        env.update({"HF_HOME": str(HOME / "hf"), "CUDA_VISIBLE_DEVICES": m.get("visible", m["gpus"]), "NCCL_NVLS_ENABLE": "0"})
        if arm in m["comps"]:
            self.kind = "comp"
            eng, extra = m["comps"][arm]
            env.update({"SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1", "PATH": f"{ENG / 'sgld' / 'bin'}:{env['PATH']}"})
            cmd = [str(ENG / "sgld" / "bin" / "sglang"), "serve", "--model-path", m["sglang_model"], "--num-gpus",
                   str(len(env["CUDA_VISIBLE_DEVICES"].split(","))), "--host", "127.0.0.1", "--port", str(port), "--scheduler-port", str(port + 1), *extra]
            cwd = str(NEWB)
        else:
            self.kind = "mstar"
            env.update({"PYTHONPATH": str(TREE), "PATH": f"{NEWB / 'venv' / 'bin'}:{env['PATH']}",
                        "MSTAR_SHM_ARENA": "1", "MSTAR_REQUIRE_ACCELERATOR_GRAPHS": "1",
                        "TORCHINDUCTOR_CACHE_DIR": str(HOME / "push4" / "cache" / m["name"] / "inductor"),
                        "TRITON_CACHE_DIR": str(HOME / "push4" / "cache" / m["name"] / "triton"),
                        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            env.update({k: v for k, v in env_extra.items() if not k.startswith("OVN_")})
            cfg = env_extra.get("OVN_CONFIG", m["config"])
            cmd = [PY, "-m", "mstar.cli.main", "serve", m["mstar"], "--config",
                   cfg if cfg.startswith("/") else str(TREE / cfg), "--gpus", m.get("visible", m["gpus"]),
                   "--port", str(port),
                   "--socket-path-prefix", f"/tmp/p4g_{port}_{int(time.time())}/"]
            cwd = str(TREE)
        self.logf = open(logpath, "w")
        self.logf.write(f"# arm={arm} env={env_extra} cmd={' '.join(cmd)}\n")
        self.logf.flush()
        self.cores = CORES if self.kind == "comp" else MSTAR_CORES
        self.p = subprocess.Popen(pin(cmd, self.cores), cwd=cwd, env=env, stdout=self.logf, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3600:
            if self.p.poll() is not None:
                raise RuntimeError(f"server {arm} died at startup; see {logpath}")
            if healthy(port):
                self.startup_s = time.monotonic() - t0
                log(f"server {arm} up in {self.startup_s:.0f}s")
                return
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"server {arm} not healthy in 60 min")

    def stop(self) -> None:
        if self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGINT)
                self.p.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait(timeout=30)
            except ProcessLookupError:
                pass
        subprocess.run(["pkill", "-KILL", "-s", str(self.p.pid)], capture_output=True)
        t0 = time.monotonic()
        while healthy(self.port) and time.monotonic() - t0 < 60:
            time.sleep(1)
        time.sleep(5)
        self.logf.close()


def cell(m: dict, kind: str, port: int, conc: int, n: int, warm_waves: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "spec.json"
    spec.write_text(json.dumps(m["spec"][kind]))
    cmd = pin([PY, "-I", str(HERE / "dgen_client.py"), "--engine", kind, "--port", str(port), "--spec", f"@{spec}",
               "--items", str(DATA / f"{m['name']}.jsonl"), "--conc", str(conc), "--num", str(n),
               "--warmup-waves", str(warm_waves), "--save-dir", str(out / "outputs"), "--out", str(out)])
    load0 = os.getloadavg()[0]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    (out / "client.log").write_text(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
    row = {"conc": conc, "n": n, "rc": r.returncode, "load0": round(load0, 1), "load1": round(os.getloadavg()[0], 1)}
    rj = out / "results.json"
    if rj.exists():
        s = json.loads(rj.read_text())["summary"]
        row.update({k: s.get(k) for k in ("completed", "failed", "throughput_per_s", "units_per_s", "lat_p50",
                                           "lat_p90", "lat_mean", "errors")})
    return row


def cmd_items(m: dict) -> None:
    from huggingface_hub import hf_hub_download
    DATA.mkdir(parents=True, exist_ok=True)
    if m["name"] == "droid":
        img = hf_hub_download("nvidia/Cosmos3-Nano", "assets/example_i2v_input.jpg")
        row = {"image_path": img, "image_b64": base64.b64encode(Path(img).read_bytes()).decode(),
               "task": "Pick up the red cup and place it in the sink.", "model_kwargs": json.dumps(DROID_MK),
               "units": 1}
        (DATA / "droid.jsonl").write_text(json.dumps(row) + "\n")
    elif m["name"] == "super_i2v":
        img = hf_hub_download(SUPER_I2V_HF, "assets/example_first_frame.png")
        prompt = json.loads(Path(hf_hub_download(SUPER_I2V_HF, "assets/example_prompt.json")).read_text())["prompt"]
        row = {"image_path": img, "prompt": prompt, "units": 189,
               "image_data_uri": "data:image/png;base64," + base64.b64encode(Path(img).read_bytes()).decode()}
        (DATA / "super_i2v.jsonl").write_text(json.dumps(row) + "\n")
    log(f"items {m['name']} -> {DATA}")


def run_arm(m, arm, env, out, r, rows_path, profile=False):
    kind = "sglang" if arm in m["comps"] else "mstar"
    if profile:
        env = {**env, "MSTAR_PHASE_TIMING": "100"}
    tag = f"{arm}_profile" if profile else f"{arm}_r{r}"
    s = Server(m, arm, env, PORT, out / "servers" / f"{tag}.log")
    try:
        # warm every batch size first (untimed), then the cells
        cell(m, kind, PORT, m["warm_conc"], m["warm_n"], 0, out / "cells" / f"{tag}_warm")
        for c in m["concs"]:
            row = cell(m, kind, PORT, c, m["nums"][c], 1, out / "cells" / f"{tag}_c{c}")
            row.update({"arm": arm, "round": r, "model": m["name"], "profile": profile, "ts": time.time(),
                        "startup_s": round(s.startup_s), "server_cores": s.cores, "client_cores": CORES})
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            log(f"{tag} c={c}: {row.get('units_per_s')} {m['units']}/s p50 {row.get('lat_p50')} "
                f"failed={row.get('failed')} load={row['load0']}->{row['load1']}")
    finally:
        s.stop()


def summarize(out: Path) -> None:
    import statistics
    rows = [json.loads(x) for x in (out / "bench.jsonl").read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        if r.get("units_per_s") and not r.get("failed") and not r.get("profile"):
            by.setdefault((r["arm"], r["conc"]), []).append(r)
    lines = ["| arm | c | units/s mean | runs | latency p50 s | load |", "|---|---:|---:|---|---:|---|"]
    for (arm, c), v in sorted(by.items()):
        u = [x["units_per_s"] for x in v]
        lines.append(f"| {arm} | {c} | {statistics.mean(u):.3f} | {[round(x, 3) for x in u]} | "
                     f"{statistics.mean(x['lat_p50'] for x in v):.3f} | {[x['load0'] for x in v]} |")
    bad = [r for r in rows if r.get("failed") or not r.get("units_per_s")]
    if bad:
        lines.append(f"\nBAD CELLS: {[(r['arm'], r['round'], r['conc'], r.get('failed')) for r in bad]}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["items", "bench", "profile", "summarize"])
    ap.add_argument("--model", default="droid")
    ap.add_argument("--arms", default="base,sglang_b1,sglang_b8,base2")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--gpu", default="", help="physical GPU(s), e.g. 2 or 2,3 (mstar --gpus / CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--after-pid", type=int, action="append", default=[],
                    help="wait for these processes (jobs ahead on the same GPUs) to exit first")
    a = ap.parse_args()
    m = {**MODELS[a.model], "name": a.model}
    if a.gpu:
        m["visible"] = a.gpu
    for pid in a.after_pid:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(20)
    if a.phase == "items":
        cmd_items(m)
        return 0
    out = Path(a.out).expanduser()
    (out / "servers").mkdir(parents=True, exist_ok=True)
    rows = out / "bench.jsonl"
    if a.phase == "summarize":
        summarize(out)
    elif a.phase == "profile":
        run_arm(m, "base", {}, out, 0, rows, profile=True)
    else:
        arms = [parse_arm(x) for x in a.arms.split(",")]
        for r in range(1, a.rounds + 1):
            for arm, env in (arms if r % 2 else arms[::-1]):
                run_arm(m, arm, env, out, r, rows)
        summarize(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
