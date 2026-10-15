"""Qwen3.5-family video (served as Qwen/Qwen3.8-27B, TP4): correctness against
transformers, then mstar vs vLLM vs SGLang, per the benchmarking skill
(mstar-project/mstar#370, `.claude/skills/benchmarking/SKILL.md`).

Stages, in the order `all` runs them; each writes under --out and is resumable:

  data         prepare_videos.py: `short` (6 s clips, all distinct) and `long`
               (whole 1-3 min Video-MME videos), Video-MME layout.
  correctness  the gate. pytest with the real tokenizer (prompt ids, grids,
               patches, positions, mp4 decode); vision tower vs transformers on
               a real video; greedy answers from mstar (two server processes:
               the same-code control), vLLM and SGLang teacher-forced through
               transformers. gate.py decides; nothing below runs on FAIL.
  sweep        `rounds` rounds; each restarts every server and rotates the
               engine order. Per point: 5 x conc requests (min 10), closed loop,
               --ignore-eos, output 1024-4096 seeded; ~5 warmup waves before a
               server's first point, 1 before later ones. A cell counts only if
               results.json has failed == 0.
  kv           the KV-pool fix's evidence: mstar at the 2048-page default vs
               the 8192 pages the sweep serves with, long video at c16/c32.
  ab           invariant 4: image_to_text on this branch vs main, from one
               checkout (branch.patch reversed / reapplied), A B A B.
  profile      mstar worker phases where it trails (short c32, long c16).
  intake       per-video decode / preprocess cost (CPU and GPU), no server.

  python bench.py all --out ~/results/video
  python bench.py summarize --out ~/results/video
  python bench.py all --out ... --dry-run          # print the plan only
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINES = ("mstar", "vllm", "sglang")
WORKLOADS = ("video_short", "video_long")


@dataclass
class Config:
    out: Path
    mstar_dir: Path = HERE.parents[1]
    bench_home: Path = Path.home() / "video_bench_envs"   # vllm/ and sglang/ venvs
    data: Path = Path.home() / "video_bench_data"
    model_id: str = "Qwen/Qwen3.8-27B"
    mstar_model: str = "qwen3_5_27b"           # the yaml points it at model_id
    mstar_config: Path = HERE / "qwen3_8_27b_tp4_kv8192.yaml"
    mstar_config_kv_default: Path = HERE / "qwen3_8_27b_tp4.yaml"
    client_model: str = "qwen3.5"
    gpus: str = "0,1,2,3"
    port: int = 8400
    engines: tuple[str, ...] = ENGINES
    workloads: tuple[str, ...] = WORKLOADS
    concurrency: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    rounds: int = 3
    first_round: int = 1
    warmup_first: int = 5
    warmup_rest: int = 1
    out_min: int = 1024
    out_max: int = 4096
    max_len: int = 32768        # the mstar yaml's max_seq_len
    max_seqs: int = 128         # gdn_state.max_slots 129 = 128 + the sink
    health_timeout: int = 3600  # a cold Inductor max-autotune at TP4 runs past 30 min
    engine_env: dict[str, dict[str, str]] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def tp(self) -> int:
        return len(self.gpus.split(","))

    @property
    def python(self) -> str:
        return str(self.mstar_dir / ".venv" / "bin" / "python")

    @property
    def server_sh(self) -> str:
        return str(HERE / "server.sh")


@dataclass
class Cell:
    engine: str
    workload: str
    conc: int
    round: int
    warmup_waves: int
    started: float = 0.0
    seconds: float = 0.0
    completed: int | None = None
    failed: int | None = None
    tok_s: float | None = None
    ttft_p50_s: float | None = None
    itl_p50_s: float | None = None
    avg_out_tokens: float | None = None
    load_before: float | None = None
    load_after: float | None = None
    log: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def loadavg() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return None


def run(cmd: list[str], cfg: Config, **kw) -> subprocess.CompletedProcess:
    print("$ " + shlex.join(cmd), flush=True)
    if cfg.dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=False, **kw)


def stage_done(cfg: Config, name: str) -> bool:
    return (cfg.out / "stages" / name).exists()


def mark_done(cfg: Config, name: str) -> None:
    (cfg.out / "stages").mkdir(parents=True, exist_ok=True)
    (cfg.out / "stages" / name).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def gpu_memory_mib(cfg: Config) -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                          "-i", cfg.gpus], capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


def wait_gpus_free(cfg: Config, timeout: float = 180) -> None:
    """Two arms never share a GPU: refuse to start while anything holds one."""
    if cfg.dry_run:
        return
    deadline = time.time() + timeout
    while (used := gpu_memory_mib(cfg)) and max(used) > 2048:
        if time.time() > deadline:
            raise RuntimeError(f"GPUs {cfg.gpus} still in use ({used} MiB); stopping rather than sharing")
        time.sleep(5)


def sglang_cache_envs(cfg: Config) -> list[str]:
    """SGLang's multimodal-cache size knobs, if the installed version has any:
    a cache hit would skip decode and the vision tower."""
    path = cfg.bench_home / "sglang_cache_envs.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []


def server_command(cfg: Config, engine: str, *, kv_default: bool = False,
                   sglang_cache_off: bool = True) -> tuple[list[str], dict[str, str]]:
    """Command and environment for one engine. Each env's bin/ goes first on
    PATH (FlashInfer's JIT shells out to that env's `ninja`)."""
    env = dict(os.environ)
    if engine == "mstar":
        env["PATH"] = f"{cfg.mstar_dir / '.venv' / 'bin'}:{env['PATH']}"
        # PYTHONPATH first, so an A/B arm imports this tree as patched now
        env.update(MSTAR_SHM_ARENA="1", MSTAR_TP_ASYNC_SCHED="1", PYTHONPATH=str(cfg.mstar_dir))
        config = cfg.mstar_config_kv_default if kv_default else cfg.mstar_config
        cmd = [cfg.python, "-m", "mstar.cli.main", "serve", cfg.mstar_model, "--config", str(config),
               "--gpus", cfg.gpus, "--port", str(cfg.port)]
    elif engine == "vllm":
        venv = cfg.bench_home / "vllm"
        env["PATH"] = f"{venv}/bin:{env['PATH']}"
        env["CUDA_VISIBLE_DEVICES"] = cfg.gpus
        # fp32 GDN state is vLLM's default; its multimodal processor cache is off
        cmd = [f"{venv}/bin/vllm", "serve", cfg.model_id, "--tensor-parallel-size", str(cfg.tp),
               "--max-model-len", str(cfg.max_len), "--max-num-seqs", str(cfg.max_seqs),
               "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0",
               "--limit-mm-per-prompt", '{"image": 1, "video": 1}', "--port", str(cfg.port)]
    elif engine == "sglang":
        venv = cfg.bench_home / "sglang"
        env["PATH"] = f"{venv}/bin:{env['PATH']}"
        env["CUDA_VISIBLE_DEVICES"] = cfg.gpus
        if sglang_cache_off:
            env.update({var: "0" for var in sglang_cache_envs(cfg)})
        cmd = [f"{venv}/bin/python", "-m", "sglang.launch_server", "--model-path", cfg.model_id,
               "--tp", str(cfg.tp), "--context-length", str(cfg.max_len),
               "--max-running-requests", str(cfg.max_seqs), "--disable-radix-cache",
               "--mamba-ssm-dtype", "float32", "--host", "0.0.0.0", "--port", str(cfg.port)]
    else:
        raise ValueError(engine)
    overrides = cfg.engine_env.get(engine, {})
    env.update(overrides)
    if "CUDA_HOME" in overrides:
        env["PATH"] = f"{overrides['CUDA_HOME']}/bin:{env['PATH']}"
    return cmd, env


def post(cfg: Config, body: dict, timeout: float = 1200) -> dict:
    req = urllib.request.Request(f"http://localhost:{cfg.port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def smoke(cfg: Config, engine: str) -> str:
    """A text request and a video request; both must answer with text before
    anything is timed. A video smoke on mstar `main` (the A/B) is skipped."""
    if cfg.dry_run:
        return "(dry run)"
    import base64
    contents = [[{"type": "text", "text": "Say hello."}]]
    if engine != "mstar_main" and (cfg.data / "short/data/items.jsonl").exists():
        row = json.loads((cfg.data / "short/data/items.jsonl").read_text().splitlines()[0])
        data = base64.b64encode((cfg.data / "short" / row["video_path"]).read_bytes()).decode()
        contents.append([{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{data}"}},
                         {"type": "text", "text": "Describe this video."}])
    texts = []
    for content in contents:
        reply = post(cfg, {"model": cfg.model_id, "max_tokens": 16, "temperature": 0.0,
                           "messages": [{"role": "user", "content": content}]})
        msg = reply["choices"][0]["message"]
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or msg.get("reasoning") or "")
        if not text.strip():
            raise RuntimeError("smoke request returned no text")
        texts.append(text[:40])
    return " | ".join(texts)


def record_versions(cfg: Config, engine: str) -> dict:
    """Which build each arm imported, so every number traces to a tree."""
    if cfg.dry_run:
        return {}
    if engine.startswith("mstar"):
        code = ("import mstar, mstar_rust, torch, flashinfer, transformers, torchcodec; "
                "print(mstar.__file__, torch.__version__, flashinfer.__version__, transformers.__version__, "
                "torchcodec.__version__)")
        env = dict(os.environ, PYTHONPATH=str(cfg.mstar_dir))
        info = subprocess.run([cfg.python, "-c", code], capture_output=True, text=True, check=False, env=env)
        video_path = (cfg.mstar_dir / "mstar/model/qwen3_5/video.py").exists()
        return {"imports": info.stdout.strip() or info.stderr.strip()[-300:], "has_video_path": video_path}
    venv = cfg.bench_home / engine
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", f"{venv}/bin/python"],
                            capture_output=True, text=True, check=False).stdout
    keep = ("vllm", "sglang", "torch=", "flashinfer", "transformers", "torchcodec")
    return {"packages": [ln for ln in freeze.splitlines() if ln.lower().startswith(keep)]}


class Server:
    """One engine served through the skill's server.sh: own process group,
    readiness against the PID it started, stop reaps the whole tree."""

    def __init__(self, cfg: Config, engine: str, tag: str, **kw):
        self.cfg, self.engine, self.tag, self.kw = cfg, engine, tag, kw
        self.name = f"bench-{engine}"

    def __enter__(self) -> "Server":
        cfg = self.cfg
        wait_gpus_free(cfg)
        serve_as = "mstar" if self.engine.startswith("mstar") else self.engine
        cmd, env = server_command(cfg, serve_as, **self.kw)
        log = cfg.out / "servers" / f"{self.engine}_{self.tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        start = run([cfg.server_sh, "start", self.name, str(cfg.port), "--timeout", str(cfg.health_timeout),
                     "--log", str(log), "--", *cmd], cfg, env=env, cwd=cfg.mstar_dir)
        if start.returncode != 0:
            run([cfg.server_sh, "stop", self.name], cfg)
            raise RuntimeError(f"{self.engine} {self.tag}: server failed to start (see {log})")
        try:
            print(f"== {self.engine} {self.tag} smoke: {smoke(cfg, self.engine)!r}", flush=True)
        except Exception:
            run([cfg.server_sh, "stop", self.name], cfg)
            raise
        if not cfg.dry_run:
            with (cfg.out / "versions.jsonl").open("a") as f:
                versions = record_versions(cfg, self.engine)
                f.write(json.dumps({"engine": self.engine, "tag": self.tag, **versions}) + "\n")
        return self

    def __exit__(self, *exc) -> None:
        run([self.cfg.server_sh, "stop", self.name], self.cfg)


def start_server(cfg: Config, engine: str, tag: str, **kw) -> Server:
    """SGLang retried without the cache knobs if they stop it from serving:
    a run with the cache on is still a run, with a note."""
    try:
        return Server(cfg, engine, tag, **kw).__enter__()
    except Exception as e:
        if engine != "sglang" or not sglang_cache_envs(cfg):
            raise
        print(f"!! sglang with {sglang_cache_envs(cfg)}=0 failed ({e!r}); retrying with its defaults", flush=True)
        (cfg.out / "notes.txt").open("a").write(f"sglang {tag}: multimodal cache knobs left at defaults\n")
        return Server(cfg, engine, tag, sglang_cache_off=False, **kw).__enter__()


def runner_args(cfg: Config, workload: str) -> list[str]:
    if workload == "video_short":
        return ["--request-type", "video_to_text", "--dataset", "video_mme", "--video-mme-dir", str(cfg.data / "short")]
    if workload == "video_long":
        return ["--request-type", "video_to_text", "--dataset", "video_mme", "--video-mme-dir", str(cfg.data / "long")]
    return ["--request-type", workload]


def bench_cell(cfg: Config, cell: Cell, out: Path) -> Cell:
    cell_dir = out / cell.workload / cell.engine / f"c{cell.conc}_{cell.round}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    n = max(10, 5 * cell.conc)
    inference = "ours_openai" if cell.engine.startswith("mstar") else "vllm_omni"
    cmd = [cfg.python, "-m", "benchmark.runner",
           "--url", f"http://localhost:{cfg.port}", "--model", cfg.client_model, "--model-id", cfg.model_id,
           "--inference-system", inference, "--profiling-type", "closed_loop", *runner_args(cfg, cell.workload),
           "--max-concurrency", str(cell.conc), "--num-requests", str(n), "--num-warmup", str(cell.warmup_waves),
           "--ignore-eos", "--output-len-min", str(cfg.out_min), "--output-len-max", str(cfg.out_max),
           "--output-dir", str(cell_dir)]
    cell.log = str(cell_dir / "bench.log")
    cell.load_before, cell.started = loadavg(), time.time()
    env = dict(os.environ, PYTHONPATH=str(cfg.mstar_dir))
    proc = run(cmd, cfg, cwd=cfg.mstar_dir, capture_output=True, text=True, env=env)
    cell.seconds, cell.load_after = time.time() - cell.started, loadavg()
    if cfg.dry_run:
        return cell
    Path(cell.log).write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        cell.notes.append(f"runner exited {proc.returncode}")
    # a run where every request failed still prints a plausible wall time
    try:
        res = json.loads((cell_dir / "results.json").read_text())
        cell.completed, cell.failed = res.get("completed"), res.get("failed")
    except (OSError, ValueError) as e:
        cell.notes.append(f"no results.json: {e}")
    sys.path.insert(0, str(cfg.mstar_dir))
    from benchmark.convenience.parse_i2t_table import find_mean_output_tokens, find_stat, find_throughput_tok
    cell.tok_s = find_throughput_tok(proc.stdout)
    cell.ttft_p50_s = find_stat(proc.stdout, "TTFT (text)", "p50")
    cell.itl_p50_s = find_stat(proc.stdout, "ITL (text)", "p50")
    cell.avg_out_tokens = find_mean_output_tokens(proc.stdout)
    return cell


def sweep_points(cfg: Config, server: Server, rnd: str, workloads, concurrency, out: Path) -> None:
    first = True
    for wl in workloads:
        for c in concurrency:
            cell = Cell(server.engine, wl, c, rnd, cfg.warmup_first if first else cfg.warmup_rest)
            first = False
            print(f"== {server.engine} {wl} conc={c} {rnd} warmup={cell.warmup_waves}", flush=True)
            cell = bench_cell(cfg, cell, out)
            if not cfg.dry_run:
                print(f"   {cell.tok_s} tok/s  ttft_p50={cell.ttft_p50_s}  failed={cell.failed}  "
                      f"{cell.seconds / 60:.1f} min {' '.join(cell.notes)}", flush=True)
                with (out / "results.jsonl").open("a") as f:
                    f.write(json.dumps(asdict(cell)) + "\n")


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_data(cfg: Config) -> None:
    r = run([cfg.python, str(HERE / "prepare_videos.py"), "--out", str(cfg.data), "--videos", "48",
             "--clips-per-video", "5"], cfg)
    if r.returncode:
        raise RuntimeError("prepare_videos failed")


def stage_correctness(cfg: Config) -> bool:
    """The gate. Every check writes its own output; gate.py reads them all."""
    out = cfg.out
    env = dict(os.environ, PYTHONPATH=str(cfg.mstar_dir))
    locate = f"from huggingface_hub import snapshot_download as d; print(d({cfg.model_id!r}))"
    ckpt = "(checkpoint dir)" if cfg.dry_run else subprocess.run(
        [cfg.python, "-c", locate], capture_output=True, text=True, check=False, env=env).stdout.strip()
    if not stage_done(cfg, "pytest"):
        r = run([cfg.python, "-m", "pytest", "-q", "-ra", "test/modular/test_qwen3_5_video.py",
                 "test/modular/test_qwen3_5_prompt_interleave.py", "test/modular/test_qwen3_5_check_stop_batched.py",
                 "test/modular/test_piecewise_can_lease.py"], cfg, cwd=cfg.mstar_dir, capture_output=True, text=True,
                env=dict(env, QWEN3_5_CKPT=ckpt, CUDA_VISIBLE_DEVICES=""))
        if not cfg.dry_run:
            (out / "pytest_real_ckpt.log").write_text(r.stdout + r.stderr)
            print((r.stdout or "").strip().splitlines()[-1:] or "(no pytest output)", flush=True)
        mark_done(cfg, "pytest")
    if not stage_done(cfg, "vision"):
        wait_gpus_free(cfg)
        items = cfg.data / "long/data/items.jsonl"
        video = (cfg.data / "long" / json.loads(items.read_text().splitlines()[0])["video_path"]
                 if not cfg.dry_run else Path("video.mp4"))
        r = run([cfg.python, str(HERE / "vision_parity.py"), "--video", str(video),
                 "--model", cfg.model_id, "--out", str(out / "vision_parity.json")], cfg, cwd=cfg.mstar_dir,
                capture_output=True, text=True, env=dict(env, CUDA_VISIBLE_DEVICES=cfg.gpus.split(",")[0]))
        if not cfg.dry_run:
            (out / "vision_parity.txt").write_text(r.stdout + r.stderr)
        mark_done(cfg, "vision")
    if not stage_done(cfg, "parity_collect"):
        # mstar twice, in two server processes: the same-code noise floor
        for engine, tag in (("mstar", "mstar_a"), ("vllm", "vllm"), ("sglang", "sglang"), ("mstar", "mstar_b")):
            try:
                server = start_server(cfg, engine, f"correctness_{tag}")
            except Exception as e:  # noqa: BLE001
                print(f"!! correctness {tag}: {e!r}", flush=True)
                continue
            try:
                run([cfg.python, str(HERE / "parity.py"), "collect", "--port", str(cfg.port), "--tag", tag,
                     "--data", str(cfg.data), "--out-dir", str(out / "parity")], cfg, cwd=cfg.mstar_dir, env=env)
            finally:
                server.__exit__()
        mark_done(cfg, "parity_collect")
    if not stage_done(cfg, "oracle"):
        wait_gpus_free(cfg)
        r = run([cfg.python, str(HERE / "parity.py"), "oracle", "--data", str(cfg.data),
                 "--out-dir", str(out / "parity")],
                cfg, cwd=cfg.mstar_dir, capture_output=True, text=True, env=dict(env, CUDA_VISIBLE_DEVICES=cfg.gpus))
        if not cfg.dry_run:
            (out / "parity").mkdir(parents=True, exist_ok=True)
            (out / "parity" / "oracle.log").write_text(r.stdout + r.stderr)
        mark_done(cfg, "oracle")
    r = run([cfg.python, str(HERE / "gate.py"), "--out", str(out)], cfg, capture_output=True, text=True)
    if not cfg.dry_run:
        (out / "gate.txt").write_text(r.stdout + r.stderr)
        print(r.stdout, flush=True)
    return r.returncode == 0


def stage_sweep(cfg: Config) -> None:
    engines = list(cfg.engines)
    for rnd in range(cfg.first_round, cfg.first_round + cfg.rounds):
        k = (rnd - 1) % len(engines)
        for engine in engines[k:] + engines[:k]:   # rotate who goes first
            if stage_done(cfg, f"sweep_{engine}_round{rnd}"):
                continue
            try:
                server = start_server(cfg, engine, f"round{rnd}")
            except Exception as e:  # noqa: BLE001 -- one engine failing must not cost the others
                print(f"!! {engine} round {rnd}: {e!r}", flush=True)
                continue
            try:
                sweep_points(cfg, server, f"round{rnd}", cfg.workloads, cfg.concurrency, cfg.out)
                mark_done(cfg, f"sweep_{engine}_round{rnd}")
            except Exception as e:  # noqa: BLE001
                print(f"!! {engine} round {rnd}: {e!r}", flush=True)
            finally:
                server.__exit__()
            if not cfg.dry_run:
                summarize(cfg.out, quiet=True)


def stage_kv(cfg: Config) -> None:
    """mstar_kv2048 against the sweep's mstar (8192 pages) at the same points."""
    out = cfg.out / "kv"
    out.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        server = Server(cfg, "mstar_kv2048", f"kv{i}", kv_default=True).__enter__()
        try:
            sweep_points(cfg, server, f"kv{i}", ("video_long",), (16, 32), out)
        finally:
            server.__exit__()


def apply_patch(cfg: Config, reverse: bool) -> None:
    flag = ["-R"] if reverse else []
    r = run(["patch", "-p1", "-s", *flag, "-i", str(HERE / "branch.patch")], cfg, cwd=cfg.mstar_dir)
    if r.returncode:
        raise RuntimeError(f"branch.patch did not {'reverse' if reverse else 'apply'} cleanly")
    # a half-switched tree dies at import; clear bytecode and check it imports
    run(["find", "mstar", "-name", "__pycache__", "-type", "d", "-prune", "-exec", "rm", "-rf", "{}", "+"], cfg,
        cwd=cfg.mstar_dir)
    r = run([cfg.python, "-c", "import mstar, mstar.conductor.conductor, mstar.worker.worker"], cfg,
            cwd=cfg.mstar_dir, env=dict(os.environ, PYTHONPATH=str(cfg.mstar_dir)))
    if r.returncode:
        raise RuntimeError("tree does not import after switching")


def stage_ab(cfg: Config) -> None:
    """image_to_text, this branch vs main (the patch reversed), A B A B back to
    back, one tree and one venv. The tree is restored whatever happens."""
    out = cfg.out / "ab"
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.dry_run:
        dry = run(["patch", "-p1", "-s", "-R", "--dry-run", "-i", str(HERE / "branch.patch")], cfg, cwd=cfg.mstar_dir)
        if dry.returncode:
            raise RuntimeError("branch.patch does not reverse cleanly onto this tree; skipping the A/B")
    reverted = False
    try:
        for i in (1, 2):
            for arm in ("mstar_main", "mstar"):
                if arm == "mstar_main" and not reverted:
                    apply_patch(cfg, reverse=True)
                    reverted = True
                elif arm == "mstar" and reverted:
                    apply_patch(cfg, reverse=False)
                    reverted = False
                server = Server(cfg, arm, f"ab{i}").__enter__()
                try:
                    sweep_points(cfg, server, f"ab{i}", ("image_to_text",), cfg.concurrency, out)
                finally:
                    server.__exit__()
    finally:
        if reverted:
            apply_patch(cfg, reverse=False)


def stage_profile(cfg: Config) -> None:
    out = cfg.out / "profile"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["MSTAR_PHASE_TIMING"] = "100"
    try:
        server = Server(cfg, "mstar", "profile").__enter__()
        try:
            log = cfg.out / "servers" / "mstar_profile.log"
            for wl, c in (("video_short", 32), ("video_long", 16)):
                start = sum(1 for _ in log.open()) if log.exists() else 0
                sweep_points(cfg, server, "profile", (wl,), (c,), out)
                end = sum(1 for _ in log.open()) if log.exists() else 0
                with (out / "marks.txt").open("a") as f:
                    f.write(f"{wl} c={c} lines {start}-{end}\n")
        finally:
            server.__exit__()
    finally:
        os.environ.pop("MSTAR_PHASE_TIMING", None)
    if not cfg.dry_run:
        code = ("import sys; from benchmark.worker_phases.phases import parse_log, render, segment; "
                "print(render(segment(parse_log(open(sys.argv[1]).read()))))")
        r = run([cfg.python, "-c", code, str(cfg.out / "servers" / "mstar_profile.log")], cfg, cwd=cfg.mstar_dir,
                capture_output=True, text=True)
        (out / "phases.txt").write_text(r.stdout + r.stderr)


def stage_intake(cfg: Config) -> None:
    wait_gpus_free(cfg)
    r = run([cfg.python, str(HERE / "video_cpu_cost.py"), "--data", str(cfg.data), "--items", "8",
             "--model", cfg.model_id], cfg, cwd=cfg.mstar_dir, capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=str(cfg.mstar_dir), CUDA_VISIBLE_DEVICES=cfg.gpus.split(",")[0]))
    if not cfg.dry_run:
        (cfg.out / "intake_cost.txt").write_text(r.stdout + r.stderr)


def run_all(cfg: Config, stages: list[str]) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "config.json").write_text(json.dumps(asdict(cfg), default=str, indent=1))
    if "data" in stages and not stage_done(cfg, "data"):
        stage_data(cfg)
        mark_done(cfg, "data")
    if "correctness" in stages:
        if not stage_correctness(cfg):
            print("CORRECTNESS GATE FAILED: nothing is benchmarked (see gate.txt)", flush=True)
            sys.exit(1)
    for name, fn in (("sweep", stage_sweep), ("kv", stage_kv), ("ab", stage_ab),
                     ("profile", stage_profile), ("intake", stage_intake)):
        if name in stages and not stage_done(cfg, name):
            try:
                fn(cfg)
                mark_done(cfg, name)
            except Exception as e:  # noqa: BLE001 -- later stages still run
                print(f"!! stage {name}: {e!r}", flush=True)
    if not cfg.dry_run:
        summarize(cfg.out)
    print("ALL DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def table(rows: list[dict], engines_order: tuple[str, ...], title: str, ref: str = "mstar") -> list[str]:
    good = [r for r in rows if r["tok_s"] and r["failed"] == 0 and not r["notes"]]
    lines: list[str] = []
    for wl in sorted({r["workload"] for r in good}):
        engines = [e for e in engines_order if any(r["engine"] == e and r["workload"] == wl for r in good)]
        others = [e for e in engines if e != ref]
        pts: dict[tuple[str, int], list[dict]] = {}
        for r in good:
            if r["workload"] == wl:
                pts.setdefault((r["engine"], r["conc"]), []).append(r)

        def tput(e, c, pts=pts):
            v = [r["tok_s"] for r in pts.get((e, c), [])]
            if not v:
                return "-", None
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            return f"{statistics.mean(v):,.0f} ±{sd:,.0f} (n={len(v)})", statistics.mean(v)

        def med_ms(e, c, key, pts=pts):
            v = [r[key] for r in pts.get((e, c), []) if r[key] is not None]
            return f"{statistics.median(v) * 1000:.0f}" if v else "-"

        lines += [f"### {title}: {wl}", ""]
        head = (["conc"] + engines + [f"{ref} vs {o}" for o in others]
                + [f"TTFT p50 ms ({' / '.join(engines)})", f"ITL p50 ms ({' / '.join(engines)})"])
        lines += ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
        for c in sorted({c for _, c in pts}):
            cells = {e: tput(e, c) for e in engines}
            row = [str(c)] + [cells[e][0] for e in engines]
            for o in others:
                m, b = cells.get(ref, ("-", None))[1], cells[o][1]
                row.append(f"**{(m / b - 1) * 100:+.1f}%**" if m and b else "-")
            row += [" / ".join(med_ms(e, c, "ttft_p50_s") for e in engines),
                    " / ".join(med_ms(e, c, "itl_p50_s") for e in engines)]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # Equal decode work is the point of --ignore-eos. mstar reports no `usage`, so
    # its count is streamed chunks and may read one higher; more is a mismatch.
    lens: dict[tuple[str, int], set] = {}
    for r in good:
        lens.setdefault((r["workload"], r["conc"]), set()).add((r["engine"], round(r["avg_out_tokens"] or 0)))
    for (wl, c), v in sorted(lens.items()):
        vals = [n for _, n in v]
        if vals and max(vals) - min(vals) > 1:
            lines.append(f"WARNING: {wl} c={c} avg output tokens differ by engine: {sorted(v)}")
    bad = [r for r in rows if r not in good]
    if bad:
        lines += ["", f"Excluded cells ({title}):", ""]
        lines += [f"- {r['engine']} {r['workload']} c={r['conc']} {r['round']}: failed={r['failed']} "
                  f"tok/s={r['tok_s']} {' '.join(r['notes'])}" for r in bad]
    return lines


def read_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []


def summarize(out: Path, quiet: bool = False) -> str:
    lines = ["Output tok/s: mean ± stdev across rounds, each round a fresh server process (n = rounds "
             "that counted; a cell counts only with failed == 0). TTFT / ITL p50: median across rounds.", ""]
    lines += table(read_rows(out / "results.jsonl"), ENGINES, "video")
    kv = read_rows(out / "kv" / "results.jsonl")
    if kv:
        sweep = [r for r in read_rows(out / "results.jsonl")
                 if r["engine"] == "mstar" and r["workload"] == "video_long" and r["conc"] in (16, 32)]
        lines += table(kv + sweep, ("mstar", "mstar_kv2048"), "KV pool: 8192 pages (sweep) vs 2048 (default)")
    ab = read_rows(out / "ab" / "results.jsonl")
    if ab:
        lines += table(ab, ("mstar", "mstar_main"), "A/B: this branch vs main, back to back")
    for name in ("gate.txt", "notes.txt"):
        if (out / name).exists():
            lines += ["", f"### {name}", "", "```", (out / name).read_text().strip(), "```"]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    if not quiet:
        print(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("all")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--stages", default="data,correctness,sweep,kv,ab,profile,intake")
    r.add_argument("--engines", default=",".join(ENGINES))
    r.add_argument("--concurrency", default="1,2,4,8,16,32")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--first-round", type=int, default=1)
    r.add_argument("--bench-home", type=Path, default=Config.bench_home)
    r.add_argument("--data", type=Path, default=Config.data)
    r.add_argument("--engine-env", action="append", default=[], metavar="ENGINE:KEY=VALUE")
    r.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "summarize":
        summarize(a.out)
        return
    engine_env: dict[str, dict[str, str]] = {}
    for item in a.engine_env:
        engine, kv = item.split(":", 1)
        key, value = kv.split("=", 1)
        engine_env.setdefault(engine, {})[key] = value
    cfg = Config(out=a.out, engines=tuple(a.engines.split(",")),
                 concurrency=tuple(int(c) for c in a.concurrency.split(",")), rounds=a.rounds,
                 first_round=a.first_round, bench_home=a.bench_home, data=a.data, engine_env=engine_env,
                 dry_run=a.dry_run)
    run_all(cfg, a.stages.split(","))


if __name__ == "__main__":
    main()
