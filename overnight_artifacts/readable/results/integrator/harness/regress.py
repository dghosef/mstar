#!/usr/bin/env python3
"""Cross-model regression suite for perf/overnight-shared (integrator agent).

Runs on the 4xH200 box. Each CELL is one model on one GPU; a cell alternates
the two arms (base, cand) with a FRESH server per arm per round (ABBA order),
then runs one MSTAR_PHASE_TIMING profile pass per arm, then correctness checks.
Cells run in parallel, one per GPU, as separate processes.

    python regress.py suite --base /root/work/arms/base --cand /root/work/arms/cand \
        --out /workspace/results/integrator/run1 [--cells qwen35,qwen3tts,zimage,kokoro] [--rounds 2]
    python regress.py cell --name qwen35 --gpu 0 --base ... --cand ... --out ...
    python regress.py report --out /workspace/results/integrator/run1

Arms are imported from their own tree via cwd + PYTHONPATH (both beat the
editable finder); every server start logs which mstar.__file__ the arm imports.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VENV = os.environ.get("REGRESS_VENV", "/root/work/mstar/.venv")
PY = f"{VENV}/bin/python"
HF_HOME = os.environ.get("HF_HOME", "/root/work/hf")

# --------------------------------------------------------------------------
# Workload definitions
# --------------------------------------------------------------------------

TOPICS = [
    "the history of the printing press", "how bridges carry load", "the water cycle",
    "the life of honey bees", "the invention of the telescope", "volcanoes and plate tectonics",
    "the economics of coffee", "how vaccines train the immune system", "the Roman road network",
    "the physics of rainbows", "the migration of whales", "how compilers optimize loops",
    "the origins of chess", "how glaciers shape valleys", "the chemistry of baking bread",
    "the design of sailing ships", "how the heart pumps blood", "the history of paper money",
    "the behavior of octopuses", "how solar panels work", "the architecture of cathedrals",
    "the discovery of penicillin", "how birds navigate", "the rules of cricket",
    "the evolution of jazz", "how tides are formed", "the Silk Road trade routes",
    "the life cycle of stars", "how electric motors work", "the culture of ancient Egypt",
    "the science of sleep", "how earthquakes are measured",
]
PROMPT_TEMPLATES = [
    "Write a detailed essay about {t}.",
    "Explain {t} to a curious high-school student, with examples.",
]
QWEN_PROMPTS = [tpl.format(t=t) for tpl in PROMPT_TEMPLATES for t in TOPICS]  # 64

GREEDY_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "List the first ten prime numbers.",
    "Explain why the sky is blue in three sentences.",
    "Write a haiku about autumn leaves.",
    "Translate 'good morning, how are you?' into Spanish and German.",
    "What is 17 multiplied by 23? Show your work.",
    "Give three tips for writing clean Python code.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]

SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the quiet river bank.",
    "Please remember to bring your umbrella, because the forecast calls for heavy rain this afternoon.",
    "Our team finished the project two weeks ahead of schedule and under budget.",
    "The museum opens at nine in the morning and closes at six in the evening.",
    "She walked along the beach, collecting smooth stones and bits of colored glass.",
    "Scientists have discovered a new species of frog in the mountains of Peru.",
    "If you follow the trail to the top of the hill, you will see the entire valley below.",
    "The train to the city leaves every twenty minutes from platform number four.",
    "He poured a cup of hot tea and sat by the window to read his favorite book.",
    "Fresh vegetables from the local market make the best soup in winter.",
    "The orchestra played a beautiful symphony that filled the concert hall with music.",
    "Learning a new language takes patience, practice, and a little bit of courage.",
    "The old lighthouse has guided sailors safely home for more than a century.",
    "Our neighbors planted a garden full of tomatoes, peppers, and sunflowers.",
    "The children laughed as the puppy chased its tail around the living room.",
    "A gentle breeze carried the scent of pine trees across the mountain lake.",
    "The library will host a free reading workshop for families next Saturday.",
    "Engineers tested the new bridge with heavy trucks before opening it to traffic.",
    "The bakery on the corner sells warm croissants every morning before sunrise.",
    "After the storm passed, a bright rainbow appeared over the green fields.",
    "Remember to save your work often so you do not lose any important changes.",
    "The astronaut described the view of the earth as calm, blue, and breathtaking.",
    "Our flight was delayed by an hour, so we had dinner at the airport.",
    "The farmer woke up early to feed the chickens and milk the cows.",
    "Thank you for calling, your appointment has been confirmed for Tuesday at noon.",
    "The painter mixed blue and yellow to create the perfect shade of green.",
    "Most people feel more productive after a good night of restful sleep.",
    "The hikers reached the summit just in time to watch the sun go down.",
    "A small cafe in the village serves the best hot chocolate in the region.",
    "The detective examined the footprints carefully before making a decision.",
    "Turn left at the next intersection and the hospital will be on your right.",
    "The students built a small robot that could sort colored blocks by size.",
]

IMAGE_PROMPTS = [
    "A red fox sitting in a snowy forest at dawn, soft golden light, photorealistic.",
    "A cozy wooden cabin beside a mountain lake, reflections on calm water, evening.",
    "A bowl of ramen with a soft-boiled egg and green onions, studio food photography.",
    "An old lighthouse on a rocky cliff during a storm, dramatic waves, cinematic.",
    "A vintage bicycle leaning against a yellow wall with flowers in a basket.",
    "A futuristic city skyline at night with neon signs and flying cars, wide shot.",
]


# --------------------------------------------------------------------------
# HTTP client (mstar /generate, NDJSON stream of {"modality","data"(b64)})
# --------------------------------------------------------------------------

async def generate(session, url, text, model_kwargs, out_mod):
    import aiohttp
    form = aiohttp.FormData()
    form.add_field("text", text)
    form.add_field("model_kwargs", json.dumps(model_kwargs))
    form.add_field("output_modalities", out_mod)
    res = {"ttft": None, "e2e": None, "chunks": 0, "error": None, "data": bytearray(), "arrivals": []}
    t0 = time.monotonic()
    try:
        async with session.post(f"{url}/generate", data=form, read_bufsize=2**24) as resp:
            if resp.status != 200:
                res["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return res
            async for raw in resp.content:
                line = raw.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not msg.get("data"):
                    continue
                mod = msg.get("modality")
                payload = base64.b64decode(msg["data"])
                if mod == "error":
                    res["error"] = payload.decode("utf-8", "replace")[:300]
                    return res
                if not payload:
                    continue
                now = time.monotonic()
                if res["ttft"] is None:
                    res["ttft"] = now - t0
                res["arrivals"].append(now)
                res["chunks"] += 1
                res["data"] += payload
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)[:300]
        return res
    res["e2e"] = time.monotonic() - t0
    return res


async def closed_loop(url, reqs, concurrency, out_mod, timeout=1800):
    """reqs: list of (text, model_kwargs). Returns (results, wall_s)."""
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    tmo = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(connector=conn, timeout=tmo) as session:
        async def one(text, kw):
            async with sem:
                return await generate(session, url, text, kw, out_mod)
        t0 = time.monotonic()
        results = await asyncio.gather(*(one(t, k) for t, k in reqs))
        wall = time.monotonic() - t0
    return results, wall


def run_closed(url, reqs, concurrency, out_mod):
    return asyncio.run(closed_loop(url, reqs, concurrency, out_mod))


def failures(results):
    return [r["error"] for r in results if r["error"]]


def pctl(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    i = (p / 100) * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def audio_sec(nbytes):
    return nbytes / (24000 * 2)


# --------------------------------------------------------------------------
# Server management
# --------------------------------------------------------------------------

def healthy(port, timeout=1.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001
        return False


def port_free(port):
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def pids_with_tag(tag):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/environ", "rb") as f:
                env = f.read()
        except OSError:
            continue
        if f"REGRESS_TAG={tag}".encode() in env.split(b"\0"):
            out.append(int(d))
    return out


def gpu_mem_used(gpu):
    gpu = str(gpu).split(",")[0]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
            text=True, timeout=30)
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


class Server:
    def __init__(self, tree, model, config, gpu, port, log_path, tag, phase_timing=None, extra_env=None):
        self.tree, self.model, self.config = tree, model, config
        self.gpu, self.port, self.log_path, self.tag = gpu, port, log_path, tag
        self.phase_timing, self.extra_env = phase_timing, extra_env or {}
        self.proc = None

    def env(self):
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": self.tree,
            "MSTAR_SHM_ARENA": "1",
            "HF_HOME": HF_HOME,
            "VIRTUAL_ENV": VENV,
            "PATH": f"{VENV}/bin:" + env.get("PATH", ""),
            "REGRESS_TAG": self.tag,
            "PYTHONUNBUFFERED": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
        })
        env.pop("MSTAR_PHASE_TIMING", None)
        if self.phase_timing:
            env["MSTAR_PHASE_TIMING"] = str(self.phase_timing)
        env.update(self.extra_env)
        return env

    def which_mstar(self):
        return subprocess.check_output(
            [PY, "-c", "import mstar; print(mstar.__file__)"], cwd=self.tree, env=self.env(), text=True).strip()

    def start(self, timeout=2400):
        for _ in range(60):
            if port_free(self.port) and not healthy(self.port):
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"port {self.port} busy before start")
        self.mem_before = gpu_mem_used(self.gpu)
        imported = self.which_mstar()
        if not imported.startswith(self.tree):
            raise RuntimeError(f"arm {self.tree} imports {imported}")
        cmd = [PY, "-m", "mstar.cli.main", "serve", self.model, "--config", f"{self.tree}/configs/{self.config}",
               "--gpus", str(self.gpu), "--port", str(self.port)]
        self.log = open(self.log_path, "w")
        self.log.write(f"# {time.strftime('%F %T')} imported={imported} cmd={' '.join(cmd)} "
                       f"phase_timing={self.phase_timing}\n")
        self.log.flush()
        self.t_start = time.monotonic()
        self.proc = subprocess.Popen(cmd, cwd=self.tree, env=self.env(), stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited {self.proc.returncode} during startup; see {self.log_path}")
            if healthy(self.port):
                self.startup_s = time.monotonic() - self.t_start
                return self.startup_s
            time.sleep(3)
        self.stop()
        raise RuntimeError(f"server not healthy after {timeout}s")

    def stop(self):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        for sig, wait in ((None, 90), (signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            if sig is not None:
                for pid in pids_with_tag(self.tag):
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
            end = time.monotonic() + wait
            while time.monotonic() < end and (self.proc.poll() is None or pids_with_tag(self.tag)):
                time.sleep(1)
            if self.proc.poll() is not None and not pids_with_tag(self.tag):
                break
        # GPU memory back to what it was before this server started?
        for _ in range(60):
            used = gpu_mem_used(self.gpu)
            if used < 0 or used <= self.mem_before + 1500:
                break
            time.sleep(2)
        self.log.close()
        self.proc = None


# --------------------------------------------------------------------------
# Phase-timing summary
# --------------------------------------------------------------------------

def phase_summary(log_path, tree, target_bs=None):
    sys.path.insert(0, tree)
    from benchmark.worker_phases.phases import parse_log, render, segment
    recs = parse_log(Path(log_path).read_text(errors="replace"))
    segs = segment(recs, skip_warmup=3, bs_tolerance=0.15, min_records=3)
    if not segs:
        segs = segment(recs, skip_warmup=0, bs_tolerance=0.5, min_records=1)
    out = {"table": render(segs), "workers": {}}
    by_w = {}
    for s in segs:
        by_w.setdefault(s.worker, []).append(s)
    for w, ss in by_w.items():
        if target_bs:
            ss = sorted(ss, key=lambda s: (abs(s.mean_bs - target_bs) > 0.25 * target_bs, -len(s.records)))
        else:
            ss = sorted(ss, key=lambda s: -len(s.records))
        best = ss[0]
        summ = best.summary()
        out["workers"][w] = {
            "bs": best.mean_bs, "records": len(best.records),
            "phases": {k: {"mean_ms": v["mean_ms"], "p50_ms": v["p50_ms"], "us_per_1k": v["us_per_1k"]}
                       for k, v in summ.items()},
        }
    return out


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------

class Cell:
    name = ""
    model = ""
    config = ""
    out_mod = "text"
    profile_bs = None
    phase_period = 100
    startup_timeout = 2400

    def __init__(self, args, gpu, port, outdir):
        self.args, self.gpu, self.port, self.out = args, gpu, port, Path(outdir)
        self.url = f"http://127.0.0.1:{port}"

    # one measured round on a live server -> dict of metrics (+ saves outputs if save_dir)
    def measure(self, save_dir):
        raise NotImplementedError

    def profile_workload(self):
        raise NotImplementedError

    def correctness(self, arm_outputs):
        return {}


def text_of(r):
    return r["data"].decode("utf-8", "replace")


class Qwen35(Cell):
    name, model, config, out_mod, profile_bs = "qwen35", "qwen3_5_0.8b", "qwen3_5_0.8b.yaml", "text", 64
    OUT = 256

    def kw(self, i, n=None, greedy_eos=False):
        k = {"max_output_tokens": n or self.OUT, "temperature": 0.0, "seed": i}
        if not greedy_eos:
            k["ignore_eos"] = True
        return k

    def reqs(self, n, start=0):
        return [(QWEN_PROMPTS[(start + i) % len(QWEN_PROMPTS)], self.kw(start + i)) for i in range(n)]

    def load(self, c, n):
        res, wall = run_closed(self.url, self.reqs(n), c, "text")
        bad = failures(res)
        toks = [r["chunks"] for r in res if not r["error"]]
        itl = [(r["e2e"] - r["ttft"]) / max(r["chunks"] - 1, 1) for r in res if not r["error"] and r["chunks"] > 1]
        return {
            "tok_s": sum(toks) / wall, "wall_s": wall, "failed": len(bad), "fail_msgs": bad[:3],
            "tokens_min": min(toks) if toks else 0, "tokens_max": max(toks) if toks else 0,
            "ttft_p50_ms": 1e3 * pctl([r["ttft"] for r in res if r["ttft"]], 50),
            "itl_mean_ms": 1e3 * statistics.fmean(itl) if itl else None,
        }

    def warmup(self):
        run_closed(self.url, self.reqs(64), 64, "text")
        run_closed(self.url, self.reqs(32), 32, "text")
        run_closed(self.url, self.reqs(2), 1, "text")

    def measure(self, save_dir):
        self.warmup()
        m = {}
        c1 = self.load(1, 6)
        m["c1_tok_s"], m["c1_itl_ms"], m["c1_ttft_ms"] = c1["tok_s"], c1["itl_mean_ms"], c1["ttft_p50_ms"]
        for c in (32, 64):
            r = self.load(c, 5 * c)
            m[f"c{c}_tok_s"], m[f"c{c}_ttft_p50_ms"], m[f"c{c}_itl_ms"] = r["tok_s"], r["ttft_p50_ms"], r["itl_mean_ms"]
            m[f"c{c}_failed"] = r["failed"]
            m[f"c{c}_tokens_range"] = [r["tokens_min"], r["tokens_max"]]
            if r["fail_msgs"]:
                m[f"c{c}_fail_msgs"] = r["fail_msgs"]
        m["c1_failed"] = c1["failed"]
        # greedy, one at a time (batching changes reduction order)
        res, _ = run_closed(self.url, [(p, self.kw(0, 128, greedy_eos=True)) for p in GREEDY_PROMPTS], 1, "text")
        greedy = [text_of(r) if not r["error"] else f"<ERROR {r['error']}>" for r in res]
        (Path(save_dir) / "greedy.json").write_text(json.dumps(greedy, indent=1))
        m["_greedy"] = greedy
        return m

    def profile_workload(self):
        self.warmup()
        return self.load(64, 320)

    def correctness(self, outs):
        # outs: {arm: [greedy lists per round]}
        b, c = outs["base"], outs["cand"]
        res = {}
        res["base_vs_cand_match"] = sum(x == y for x, y in zip(b[0], c[0], strict=False))
        res["n"] = len(b[0])
        if len(b) > 1:
            res["base_r0_vs_r1_match"] = sum(x == y for x, y in zip(b[0], b[1], strict=False))
        if len(c) > 1:
            res["cand_r0_vs_r1_match"] = sum(x == y for x, y in zip(c[0], c[1], strict=False))
        # all rounds pooled: does cand produce any text base never produced?
        bset = [set(col) for col in zip(*b, strict=False)]
        res["cand_outputs_seen_in_base"] = sum(any(x in bset[i] for x in col)
                                               for i, col in enumerate(zip(*c, strict=False)))
        res["first_diff"] = None
        for i, (x, y) in enumerate(zip(b[0], c[0], strict=False)):
            if x != y:
                pre = os.path.commonprefix([x, y])
                res["first_diff"] = {"prompt": i, "common_prefix_chars": len(pre),
                                     "base": x[len(pre):len(pre) + 80], "cand": y[len(pre):len(pre) + 80]}
                break
        return res


class AudioCell(Cell):
    out_mod = "audio"
    conc = 16
    n_measure = 80
    kwargs = {}

    def kw(self, i):
        return {**self.kwargs, "seed": i}

    def reqs(self, n):
        return [(SENTENCES[i % len(SENTENCES)], self.kw(i)) for i in range(n)]

    def load(self, c, n, save_dir=None):
        res, wall = run_closed(self.url, self.reqs(n), c, "audio")
        bad = failures(res)
        ok = [r for r in res if not r["error"]]
        secs = sum(audio_sec(len(r["data"])) for r in ok)
        rtf = [r["e2e"] / max(audio_sec(len(r["data"])), 1e-6) for r in ok]
        if save_dir is not None:
            d = Path(save_dir) / "wav"
            d.mkdir(parents=True, exist_ok=True)
            for i, r in enumerate(res[:len(SENTENCES)]):
                if not r["error"]:
                    (d / f"{i:02d}.pcm").write_bytes(bytes(r["data"]))
        return {
            "audio_s_per_s": secs / wall, "wall_s": wall, "failed": len(bad), "fail_msgs": bad[:3],
            "ttfa_p50_ms": 1e3 * pctl([r["ttft"] for r in ok], 50) if ok else None,
            "rtf_p50": pctl(rtf, 50), "audio_s_total": secs,
        }

    def warmup(self):
        run_closed(self.url, self.reqs(self.conc), self.conc, "audio")
        run_closed(self.url, self.reqs(2), 1, "audio")

    def measure(self, save_dir):
        self.warmup()
        r = self.load(self.conc, self.n_measure, save_dir=save_dir)
        c1 = self.load(1, 4)
        m = {f"c{self.conc}_audio_s_per_s": r["audio_s_per_s"], f"c{self.conc}_ttfa_p50_ms": r["ttfa_p50_ms"],
             f"c{self.conc}_rtf_p50": r["rtf_p50"], f"c{self.conc}_failed": r["failed"],
             f"c{self.conc}_audio_s_total": r["audio_s_total"],
             "c1_ttfa_p50_ms": c1["ttfa_p50_ms"], "c1_rtf_p50": c1["rtf_p50"], "c1_failed": c1["failed"]}
        if r["fail_msgs"]:
            m["fail_msgs"] = r["fail_msgs"]
        return m

    def profile_workload(self):
        self.warmup()
        return self.load(self.conc, 3 * self.conc)

    def correctness(self, outs):
        import numpy as np
        res = {}
        # WER per arm on round-0 wavs (whisper on this cell's GPU, servers stopped)
        dirs = {arm: self.out / f"{arm}_r0" / "wav" for arm in ("base", "cand")}
        try:
            res.update(wer_dirs(dirs, self.gpu))
        except Exception as e:  # noqa: BLE001
            res["wer_error"] = repr(e)[:300]
        # waveform agreement (meaningful for deterministic models only)
        agree, lens = [], []
        for i in range(len(SENTENCES)):
            pb, pc = dirs["base"] / f"{i:02d}.pcm", dirs["cand"] / f"{i:02d}.pcm"
            if not (pb.exists() and pc.exists()):
                continue
            ra, rb = pb.read_bytes(), pc.read_bytes()
            a = np.frombuffer(ra[:len(ra) // 2 * 2], np.int16).astype(np.float32)
            b = np.frombuffer(rb[:len(rb) // 2 * 2], np.int16).astype(np.float32)
            lens.append(len(a) == len(b))
            n = min(len(a), len(b))
            if n:
                err = np.mean((a[:n] - b[:n]) ** 2)
                sig = np.mean(a[:n] ** 2) + 1e-9
                agree.append(10 * np.log10(sig / max(err, 1e-9)))
        res["same_length_frac"] = sum(lens) / max(len(lens), 1)
        res["waveform_snr_db_median"] = float(np.median(agree)) if agree else None
        return res


class Qwen3TTS(AudioCell):
    name, model, config, profile_bs = "qwen3tts", "qwen3_tts", "qwen3tts.yaml", 16
    conc, n_measure = 16, 80
    phase_period = 50
    kwargs = {"voice": "vivian", "language": "English"}


class Kokoro(AudioCell):
    name, model, config, profile_bs = "kokoro", "kokoro", "kokoro.yaml", 32
    conc, n_measure = 32, 160
    phase_period = 10
    kwargs = {"voice": "af_heart"}


class Qwen3OmniT2S(AudioCell):
    """Qwen3-Omni (configs/qwen3omni_2gpu.yaml, 2 GPUs) text -> speech, read-aloud prompts."""
    name, model, config, profile_bs = "omni", "qwen3_omni", "qwen3omni_2gpu.yaml", 8
    conc, n_measure = 8, 32
    phase_period = 50
    startup_timeout = 3600
    kwargs = {"max_tokens": 128, "max_output_tokens": 128, "temperature": 0.0}

    def reqs(self, n):
        return [(f"Read this sentence aloud exactly, and say nothing else: {SENTENCES[i % len(SENTENCES)]}",
                 self.kw(i)) for i in range(n)]


class ZImage(Cell):
    name, model, config, out_mod, profile_bs = "zimage", "z_image_turbo", "z_image_turbo.yaml", "image", 1
    phase_period = 5

    def reqs(self, idx):
        return [(IMAGE_PROMPTS[i], {"height": 1024, "width": 1024, "seed": 1000 + i}) for i in idx]

    def measure(self, save_dir):
        run_closed(self.url, self.reqs([0, 1]), 1, "image")  # warmup
        lat, bad = [], []
        d = Path(save_dir) / "img"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(len(IMAGE_PROMPTS)):
            res, wall = run_closed(self.url, self.reqs([i]), 1, "image")
            r = res[0]
            if r["error"]:
                bad.append(r["error"])
                continue
            lat.append(r["e2e"])
            (d / f"{i}.png").write_bytes(bytes(r["data"]))
        # concurrency 4 throughput (batched DiT path)
        res, wall = run_closed(self.url, self.reqs([0, 1, 2, 3] * 2), 4, "image")
        m = {"lat_mean_s": statistics.fmean(lat) if lat else None, "lat_p50_s": pctl(lat, 50),
             "lat_min_s": min(lat) if lat else None, "failed": len(bad) + len(failures(res)),
             "c4_img_per_s": sum(1 for r in res if not r["error"]) / wall}
        if bad:
            m["fail_msgs"] = bad[:3]
        return m

    def profile_workload(self):
        run_closed(self.url, self.reqs([0]), 1, "image")
        res, wall = run_closed(self.url, self.reqs([1, 2, 3]), 1, "image")
        return {"wall_s": wall}

    def correctness(self, outs):
        import numpy as np
        from PIL import Image

        def load(p):
            return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)

        def psnr(a, b):
            mse = np.mean((a - b) ** 2)
            return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

        res = {}
        for tag, (x, y) in {"base_vs_cand_r0": ("base_r0", "cand_r0"), "base_r0_vs_r1": ("base_r0", "base_r1"),
                            "cand_r0_vs_r1": ("cand_r0", "cand_r1")}.items():
            vals = []
            for i in range(len(IMAGE_PROMPTS)):
                p, q = self.out / x / "img" / f"{i}.png", self.out / y / "img" / f"{i}.png"
                if p.exists() and q.exists():
                    vals.append(psnr(load(p), load(q)))
            if vals:
                res[f"psnr_{tag}_min"] = min(vals)
                res[f"psnr_{tag}_median"] = float(np.median(vals))
        return res


class Qwen35DP2(Qwen35):
    """Two data-parallel replicas of the whole Qwen3.5-0.8B deployment (conductor replica choice)."""
    name, config, profile_bs = "qwen35dp2", "ovn_qwen3_5_0.8b_dp2.yaml", 32


CELLS = {c.name: c for c in (Qwen35, Qwen35DP2, Qwen3TTS, Qwen3OmniT2S, ZImage, Kokoro)}


# --------------------------------------------------------------------------
# WER
# --------------------------------------------------------------------------

def _norm(s):
    import re
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def wer_dirs(dirs, gpu):
    import jiwer
    import numpy as np
    import torch
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                    torch_dtype=torch.float16, device=f"cuda:{str(gpu).split(',')[0]}")
    out = {}
    for arm, d in dirs.items():
        refs, hyps = [], []
        for i in range(len(SENTENCES)):
            p = d / f"{i:02d}.pcm"
            if not p.exists():
                continue
            raw = p.read_bytes()
            a = np.frombuffer(raw[:len(raw) // 2 * 2], np.int16).astype(np.float32) / 32768.0
            txt = pipe({"raw": a, "sampling_rate": 24000}, generate_kwargs={"language": "english"})["text"]
            refs.append(_norm(SENTENCES[i]))
            hyps.append(_norm(txt))
        out[f"wer_{arm}"] = jiwer.wer(refs, hyps) if refs else None
        out[f"wer_{arm}_n"] = len(refs)
    del pipe
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------

def run_cell(args):
    cls = CELLS[args.name]
    outdir = Path(args.out) / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    port = args.port
    cell = cls(args, args.gpu, port, outdir)
    arms = {"base": args.base, "cand": args.cand}
    rounds = args.rounds
    result = {"cell": args.name, "gpu": args.gpu, "arms": arms, "rounds": [], "profile": {}, "errors": []}
    for f in ("base", "cand"):
        rev = Path(arms[f]) / ".arm_rev"
        result[f"{f}_rev"] = rev.read_text().strip() if rev.exists() else "?"
    outs = {"base": [], "cand": []}

    def save():
        (outdir / "result.json").write_text(json.dumps(result, indent=1, default=str))

    for r in range(rounds):
        order = ["base", "cand"] if r % 2 == 0 else ["cand", "base"]
        for arm in order:
            tag = f"{args.name}-{arm}-r{r}-{os.getpid()}"
            sd = outdir / f"{arm}_r{r}"
            sd.mkdir(exist_ok=True)
            srv = Server(arms[arm], cls.model, cls.config, args.gpu, port, str(sd / "server.log"), tag)
            entry = {"round": r, "arm": arm, "t": time.strftime("%T"), "load1": os.getloadavg()[0]}
            try:
                entry["startup_s"] = srv.start(cls.startup_timeout)
                # shared hosts: don't measure under heavy cross-tenant load (wait up to 10 min)
                for _ in range(60):
                    if os.getloadavg()[0] <= 250:
                        break
                    time.sleep(10)
                entry["load1_measure"] = os.getloadavg()[0]
                m = cell.measure(sd)
                greedy = m.pop("_greedy", None)
                if greedy is not None:
                    outs[arm].append(greedy)
                entry["metrics"] = m
            except Exception as e:  # noqa: BLE001
                entry["error"] = repr(e)[:500]
                result["errors"].append(f"{arm} r{r}: {e!r}"[:500])
            finally:
                srv.stop()
            entry["load1_end"] = os.getloadavg()[0]
            result["rounds"].append(entry)
            print(f"[{args.name}] {arm} r{r}: {json.dumps(entry, default=str)[:400]}", flush=True)
            save()
    if not args.no_profile:
        order = ["base", "cand"] if rounds % 2 == 0 else ["cand", "base"]
        for arm in order:
            tag = f"{args.name}-{arm}-prof-{os.getpid()}"
            sd = outdir / f"{arm}_prof"
            sd.mkdir(exist_ok=True)
            log = sd / "server.log"
            srv = Server(arms[arm], cls.model, cls.config, args.gpu, port, str(log), tag, phase_timing=cls.phase_period)
            try:
                srv.start(cls.startup_timeout)
                load = cell.profile_workload()
                srv.stop()
                ps = phase_summary(log, arms["base"], cls.profile_bs)
                (sd / "phases.txt").write_text(ps.pop("table"))
                result["profile"][arm] = {"load": load, **ps}
            except Exception as e:  # noqa: BLE001
                result["errors"].append(f"{arm} profile: {e!r}"[:500])
            finally:
                srv.stop()
            save()
    try:
        result["correctness"] = cell.correctness(outs)
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"correctness: {e!r}"[:500])
    save()
    summarize_cell(result, outdir)
    return 0


KEY_PHASES = ["iter_total", "worker.postprocess.event_sync", "worker.gpu_thread.prepare_inputs",
              "worker.gpu_thread.exec", "await_gpu", "worker.postprocess_batch", "worker.schedule",
              "worker.plan_thread.pre_plan", "submit_spec"]


def summarize_cell(result, outdir):
    lines = [f"## {result['cell']} (GPU {result['gpu']})  base={result.get('base_rev')}  cand={result.get('cand_rev')}"]
    per = {"base": {}, "cand": {}}
    for e in result["rounds"]:
        for k, v in (e.get("metrics") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                per[e["arm"]].setdefault(k, []).append(v)
    summary = {}
    lines.append("| metric | base (rounds) | cand (rounds) | delta |")
    lines.append("| --- | --- | --- | --- |")
    for k in sorted(set(per["base"]) | set(per["cand"])):
        b, c = per["base"].get(k, []), per["cand"].get(k, [])
        bm = statistics.fmean(b) if b else None
        cm = statistics.fmean(c) if c else None
        d = (cm - bm) / bm * 100 if bm and cm is not None else None
        summary[k] = {"base": b, "cand": c, "delta_pct": d}
        fmt = lambda xs: ", ".join(f"{x:.4g}" for x in xs)  # noqa: E731
        lines.append(f"| {k} | {fmt(b)} | {fmt(c)} | {'' if d is None else f'{d:+.2f}%'} |")
    result["summary"] = summary
    if result.get("profile"):
        lines.append("\nProfile (MSTAR_PHASE_TIMING=100, steady segment, mean_ms):")
        workers = sorted({w for a in result["profile"].values() for w in a.get("workers", {})})
        for w in workers:
            pb = result["profile"].get("base", {}).get("workers", {}).get(w, {})
            pc = result["profile"].get("cand", {}).get("workers", {}).get(w, {})
            lines.append(f"- {w}: bs base={pb.get('bs', 0):.1f} cand={pc.get('bs', 0):.1f}")
            for ph in KEY_PHASES:
                x = pb.get("phases", {}).get(ph, {}).get("mean_ms")
                y = pc.get("phases", {}).get(ph, {}).get("mean_ms")
                if x is not None or y is not None:
                    lines.append(f"    {ph}: {x if x is None else f'{x:.3f}'} -> {y if y is None else f'{y:.3f}'}")
    if result.get("correctness"):
        lines.append(f"\nCorrectness: {json.dumps(result['correctness'], default=str)}")
    if result["errors"]:
        lines.append(f"\nERRORS: {result['errors']}")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    (outdir / "result.json").write_text(json.dumps(result, indent=1, default=str))


def run_suite(args):
    """One worker thread per GPU pulls cells off a queue; cells on one GPU run one after another."""
    import queue
    import threading
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in str(args.gpus).split(",")]
    q = queue.Queue()
    for k, name in enumerate(args.cells.split(",")):
        q.put((k, name))

    def worker(gpu):
        while True:
            try:
                k, name = q.get_nowait()
            except queue.Empty:
                return
            cmd = [sys.executable, os.path.abspath(__file__), "cell", "--name", name, "--gpu", str(gpu),
                   "--port", str(args.port + 10 * k), "--base", args.base, "--cand", args.cand, "--out", args.out,
                   "--rounds", str(args.rounds_for.get(name, args.rounds))]
            if args.no_profile:
                cmd.append("--no-profile")
            with open(out / f"{name}.driver.log", "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"cell {name} gpu {gpu} exit {rc}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report(args)
    (out / "SUITE_DONE").write_text(time.strftime("%F %T"))


def report(args):
    out = Path(args.out)
    parts = [f"# Regression suite {out.name}  ({time.strftime('%F %T')})\n"]
    for d in sorted(out.iterdir()):
        s = d / "summary.md"
        if d.is_dir() and s.exists():
            parts.append(s.read_text())
    (out / "SUMMARY.md").write_text("\n".join(parts))
    print("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("suite", "cell", "report"):
        p = sub.add_parser(n)
        p.add_argument("--out", required=True)
        if n != "report":
            p.add_argument("--base", default=os.environ.get("REGRESS_ARMS", "/root/work/arms") + "/base")
            p.add_argument("--cand", default=os.environ.get("REGRESS_ARMS", "/root/work/arms") + "/cand")
            p.add_argument("--gpu", default="0", help="GPU index, or a comma list for multi-GPU cells")
            p.add_argument("--port", type=int, default=9100)
            p.add_argument("--rounds", type=int, default=2)
            p.add_argument("--no-profile", action="store_true")
        if n == "suite":
            p.add_argument("--cells", default="qwen35,qwen3tts,zimage,kokoro")
            p.add_argument("--gpus", default="0,1,2,3")
            p.add_argument("--rounds-per-cell", default="", help="name:n,... overrides --rounds")
        if n == "cell":
            p.add_argument("--name", required=True)
    args = ap.parse_args()
    if args.cmd == "suite":
        args.rounds_for = {k: int(v) for k, v in (x.split(":") for x in args.rounds_per_cell.split(",") if x)}
        return run_suite(args)
    if args.cmd == "cell":
        return run_cell(args)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
