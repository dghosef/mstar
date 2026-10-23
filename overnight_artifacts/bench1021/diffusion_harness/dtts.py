#!/usr/bin/env python3
"""TTS entries of the node2 queue (dqueue.py loads this per phase): M* (PR d248477ec, main abdfd5802) vs
the engines that serve each model, streaming /v1/audio/speech-style APIs, one client (dtts_client.py).

Models and arms (research notes in REPORT_TTS.md):
  qwen3tts_1p7b / qwen3tts_0p6b  Qwen3-TTS 12Hz CustomVoice, voice vivian, English, 24 kHz:
                                 vllm-omni 0.30.0 (deploy YAMLs), SGLang-Omni 0.1.9
  fish_s2pro                     Fish S2-Pro, no reference voice, 44.1 kHz: SGLang-Omni 0.1.9, vllm-omni 0.30.0,
                                 fish-speech official api_server (v3.0.0-beta; batch 1, serial)
  orpheus                        Orpheus 3B 0.1 ft, voice tara, 24 kHz: vox-serve 0.1.0, and a thin vLLM 0.30.0
                                 AsyncLLM + batched-SNAC server (orpheus_vllm_server.py: orpheus-speech's decode
                                 math on modern vLLM; no public vLLM-based Orpheus server serves concurrently);
                                 M* single-GPU colocated config; M* frontend python vs rust tuned (adapter ported)
  kokoro                         Kokoro-82M, voice af_heart, 24 kHz: Kokoro-FastAPI v0.9.0 (remsky, from source;
                                 serializes requests per process, so 1 and 4 uvicorn workers are tuned)
  chatterbox                     Chatterbox (base), built-in voice, temp 0.8 / exaggeration 0.5 / cfg 0.5, watermark
                                 off: chatterbox-vllm @8630314 (offline batch, no server) and Chatterbox-TTS-Server
                                 @915ae28 (HTTP around the official package, serial, voice Abigail.wav, no streaming)
  omnivoice                      OmniVoice, auto voice, 32 steps, guidance 2.0, 24 kHz: vllm-omni 0.30.0 (serves it;
                                 not streamed) and the official omnivoice package + FlashInfer (offline batch)
Offline-batch arms (no server): the package's own batch inference, batch b in the c column, TTFA = batch latency.

Phases: setup (idempotent installs + per-engine smoke; an engine whose install or smoke fails is dropped
and logged), tune (competitor knob sets and, where the route is ported, M*'s frontend; each knob set
also collects its gate WAVs), collect (M* PR twice, M* main), gate (WER), select, rounds, profile,
summarize.

Metrics per cell: audio-s/s (total audio seconds / wall), RTF p50 / mean, TTFA p50 / p99, failed.
Points: c = 1, 2, 4, 8, 16, 32, 64 (fish-speech official: 1, 2, 4, 8, it serves one request at a time).

Gate (criteria fixed before any collection): 128 LibriSpeech test-clean sentences per arm, synthesized
at c=8 (untimed) with the model's default sampling; Whisper large-v3 transcripts, Whisper English
normalizer, corpus WER. spread = |WER(mstar_pr) - WER(mstar_pr_b)| (two M* PR processes).
An arm PASSES if (1) WER <= best WER among all arms + max(1.0, 2 x spread) points, (2) its median
seconds-per-word is within +-25% of the median over all arms, (3) no failed / silent / runaway output.
"""
from __future__ import annotations

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
from dbench import ENG, HERE, PORT0, RESULTS, STATUS, TREES, WORK, failure, log, loadavg, reap_all, tail

TEXTS = WORK / "data/libri_tts256.txt"
TEXTS_JSONL = WORK / "data/libri_tts256.jsonl"   # the same sentences for the offline arms (item i = line i)
CONCS = (1, 2, 4, 8, 16, 32, 64)
NUMS = {1: 16, 2: 16, 4: 32, 8: 48, 16: 80, 32: 160, 64: 256}
TUNE = ((1, 16), (32, 160))
GATE_N = 128
VOSP = ENG / "vllmomni/lib/python3.12/site-packages/vllm_omni"
CFGDIR = HERE / "tts_configs"

TTS = {
    "qwen3tts_1p7b": dict(hf="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", mstar="qwen3_tts_1p7b",
                          config="configs/qwen3tts_1p7b.yaml", sr=24000, voice="vivian", language="English",
                          comps=["vllmomni", "sgomni"], sg_cfg="Qwen3TTSPipelineConfig", frontend=False),
    "qwen3tts_0p6b": dict(hf="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", mstar="qwen3_tts", config="configs/qwen3tts.yaml",
                          sr=24000, voice="vivian", language="English", comps=["vllmomni", "sgomni"],
                          sg_cfg="Qwen3TTSPipelineConfig", frontend=False),
    "fish_s2pro": dict(hf="fishaudio/s2-pro", mstar="fish_s2_pro", config="configs/fish_s2_pro.yaml", sr=44100,
                       voice=None, language=None, comps=["sgomni", "vllmomni", "fishspeech"],
                       sg_cfg="S2ProPipelineConfig", frontend=False,
                       sampling={"temperature": 0.8, "top_p": 0.8, "top_k": 30, "repetition_penalty": 1.0}),
    "orpheus": dict(hf="canopylabs/orpheus-3b-0.1-ft", mstar="orpheus", config="configs/orpheus_colocated.yaml",
                    sr=24000, voice="tara", language=None, comps=["voxserve", "orpheusvllm"], frontend=True),
    "kokoro": dict(hf="hexgrad/Kokoro-82M", mstar="kokoro", config="configs/kokoro.yaml", sr=24000, voice="af_heart",
                   language=None, comps=["kokorofastapi"], frontend=False, mstar_body={"speed": 1.0},
                   overlay=["en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
                            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"]),
    "chatterbox": dict(hf="ResembleAI/chatterbox", mstar="chatterbox", config="configs/chatterbox.yaml", sr=24000,
                       voice="default", language=None, comps=["cbvllm", "cbserver"], frontend=False,
                       mstar_body={"temperature": 0.8, "exaggeration": 0.5, "cfg_weight": 0.5, "watermark": False},
                       overlay=["pyloudnorm==0.1.1"]),
    "omnivoice": dict(hf="k2-fsa/OmniVoice", mstar="omnivoice", config="configs/omnivoice.yaml", sr=24000, voice=None,
                      language=None, comps=["vllmomni", "ovofficial"], frontend=False,
                      mstar_body={"num_step": 32, "guidance_scale": 2.0},
                      overlay=["omnivoice @ git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2",
                               "pydub==0.25.1"]),
}
OVERLAY = WORK / "overlays"   # per-model pure-Python extras for the M* arms (the shared M* venvs stay untouched)
# offline-batch arms: the package's own batch inference (dgen's doff_common contract), one process per pass
OFFLINE = {
    "cbvllm": dict(python=str(ENG / "chatterbox-vllm/.venv/bin/python"), script="off_cbvllm.py", cwd=str(ENG / "chatterbox-vllm"),
                   env={"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CHATTERBOX_CFG_SCALE": "0.5"},
                   batches=(1, 8, 32, 64), git=str(ENG / "chatterbox-vllm")),
    "ovofficial": dict(python=str(WORK / "mstar/.venv/bin/python"), script="off_omnivoice.py", cwd=str(WORK),
                       env={"DGEN_OVERLAY": str(WORK / "overlays/omnivoice")}, batches=(1, 8, 32, 64)),
}

# Knob sets per engine and model family. vllm-omni's are its own deploy configs; the names are what the
# tune table reports.
def knob_table(key, eng):
    t = TTS[key]
    if eng in TREES:
        return {"python": [], "rust": ["--rust-frontend"]} if t["frontend"] else {"python": []}
    if eng == "vllmomni":
        if key == "omnivoice":
            return {"default": ["--deploy-config", str(VOSP / "deploy/omnivoice.yaml"), "--trust-remote-code"],
                    "fp16_mns32": ["--deploy-config", str(CFGDIR / "vllmomni_omnivoice_fp16_mns32.yaml"),
                                   "--trust-remote-code"]}
        if key.startswith("qwen3tts"):
            return {"default": ["--deploy-config", str(VOSP / "deploy/qwen3_tts.yaml")],
                    "high_conc": ["--deploy-config", str(VOSP / "deploy/qwen3_tts_high_concurrency_mrv2_single_gpu.yaml")]}
        return {"high_conc": ["--deploy-config", str(CFGDIR / "vllmomni_fish_hc_topp08.yaml")],
                "default": ["--deploy-config", str(CFGDIR / "vllmomni_fish_default_topp08.yaml")]}
    if eng == "sgomni":
        base = ["--config", str(CFGDIR / f"sgomni_{key}.yaml")]
        return {"base": base,
                "mrr128": base + ["--tts_engine.engine.max_running_requests", "128",
                                  "--tts_engine.engine.max_queued_requests", "512"]}
    if eng == "voxserve":
        return {"bs64": ["--max-batch-size", "64"], "bs64_compile": ["--max-batch-size", "64", "--enable-torch-compile"],
                "bs64_async": ["--max-batch-size", "64", "--async-scheduling"]}
    if eng == "fishspeech":
        return {"compile": ["--compile"], "compile_half": ["--compile", "--half"]}
    if eng == "orpheusvllm":
        return {"default": [], "ignore_seed": ["--ignore-seed"]}
    if eng == "kokorofastapi":
        return {"workers1": [], "workers4": ["--workers", "4"]}
    if eng == "cbserver":
        return {"default": []}
    if eng == "cbvllm":  # the port's default is eager (its README: CUDA graphs "causing correctness issues")
        return {"eager": [], "graphs": ["--compile"]}
    if eng == "ovofficial":
        return {"flashinfer": [], "flashinfer_graph": ["--cuda-graph"]}
    raise ValueError(eng)


def label(eng):
    return {"mstar-pr": "M* PR", "mstar-main": "M* main", "vllmomni": "vllm-omni 0.30.0", "sgomni": "SGLang-Omni 0.1.9",
            "voxserve": "vox-serve 0.1.0", "fishspeech": "fish-speech v3.0.0-beta (official)",
            "orpheusvllm": "vLLM 0.30.0 + SNAC (thin server)", "kokorofastapi": "Kokoro-FastAPI v0.9.0",
            "cbserver": "Chatterbox-TTS-Server @915ae28 (official package, serial)",
            "cbvllm": "chatterbox-vllm @8630314 (offline batch)",
            "ovofficial": "omnivoice official + FlashInfer (offline batch)"}.get(eng, eng)


class TTSBench(dbench.Bench):
    def __init__(self, key):
        dbench.MODELS.setdefault(key, {"kind": "tts", "hf": TTS[key]["hf"], "steps": None})
        super().__init__(key)
        self.t = TTS[key]

    # ------------------------------------------------------------ engines
    def health_path(self, eng):
        return {"fishspeech": "/v1/health", "cbserver": "/docs"}.get(eng, "/health")

    def server_cmd(self, arm, gpu, port, args, env_extra=None):
        t = self.t
        env = dict(os.environ)
        env.update(HF_HOME=str(WORK / "hf"), PATH=f"/usr/local/bin:{env['PATH']}")
        tok = WORK / "hf/token"
        if tok.exists():
            env["HF_TOKEN"] = tok.read_text().strip()
        if arm in TREES:
            tr = TREES[arm]
            env["PATH"] = f"{tr}/.venv/bin:{env['PATH']}"
            env.update(MSTAR_SHM_ARENA="1", TORCHINDUCTOR_CACHE_DIR=str(WORK / f"inductor/{arm}"))
            if t.get("overlay"):
                env["PYTHONPATH"] = str(OVERLAY / self.key)
            cmd = [f"{tr}/.venv/bin/python", "-m", "mstar.cli.main", "serve", t["mstar"], "--config",
                   str(tr / t["config"]), "--gpus", str(gpu), "--port", str(port)] + list(args)
            cwd = tr
        elif arm == "vllmomni":
            v = ENG / "vllmomni"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [f"{v}/bin/vllm", "serve", t["hf"], "--omni", "--host", "127.0.0.1", "--port", str(port)] + list(args)
            cwd = ENG
        elif arm == "sgomni":
            v = ENG / "sgomni"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [f"{v}/bin/sgl-omni", "serve", "--model-path", t["hf"], "--host", "127.0.0.1", "--port", str(port)] + \
                list(args)
            cwd = ENG
        elif arm == "voxserve":
            v = ENG / "voxserve"
            env["PATH"] = f"{v}/bin:{env['PATH']}"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [f"{v}/bin/vox-serve", "--model", t["hf"], "--host", "127.0.0.1", "--port", str(port)] + list(args)
            cwd = ENG
        elif arm == "orpheusvllm":
            v = ENG / "orpheusvllm"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [f"{v}/bin/python", str(HERE / "orpheus_vllm_server.py"), "--host", "127.0.0.1", "--port", str(port),
                   "--model", t["hf"], "--tokenizer", t["hf"], "--edges", "full", "--max-model-len", "4096",
                   "--gpu-memory-utilization", "0.85"] + list(args)
            cwd = ENG
        elif arm == "kokorofastapi":
            k = ENG / "kokoro-fastapi"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), USE_GPU="true", DEVICE_TYPE="cuda", PYTHONPATH=f"{k}:{k}/api",
                       MODEL_DIR="src/models", VOICES_DIR="src/voices/v1_0", WEB_PLAYER_PATH=f"{k}/web",
                       ENABLE_WEB_PLAYER="false")
            cmd = [str(k / ".venv/bin/uvicorn"), "api.src.main:app", "--host", "127.0.0.1", "--port", str(port)] + list(args)
            cwd = k
        elif arm == "cbserver":  # host/port and generation defaults live in its config.yaml (setup writes them)
            c = ENG / "cbserver"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [str(c / ".venv/bin/python"), "server.py"] + list(args)
            cwd = c
        elif arm == "fishspeech":
            fs = ENG / "fish-speech"
            env.update(CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [str(fs / ".venv/bin/python"), "tools/api_server.py", "--listen", f"127.0.0.1:{port}",
                   "--llama-checkpoint-path", "checkpoints/s2-pro", "--decoder-checkpoint-path",
                   "checkpoints/s2-pro/codec.pth", "--decoder-config-name", "modded_dac_vq"] + list(args)
            cwd = fs
        else:
            raise ValueError(arm)
        if env_extra:
            env.update(env_extra)
        return cmd, env, cwd

    def start(self, arm, label_, gpu=0, port=PORT0, args=(), env_extra=None, name=None):
        name = name or f"t{port}"
        cmd, env, cwd = self.server_cmd(arm, gpu, port, args, env_extra)
        slog = self.out / "servers" / f"{label_}.log"
        (self.out / "servers" / f"{label_}.cmd").write_text(shlex.join(cmd) + "\n")
        limit = 3600 if arm in TREES else 2400
        log(f"start {label_} gpu{gpu}:{port}: {shlex.join(cmd)}")
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

    def spec(self, arm):
        t = self.t
        if arm in TREES:
            body = {"model": t["hf"], "input": "{text}", "response_format": "wav", "stream": True, "seed": "{seed}",
                    **t.get("mstar_body", {})}
            fmt = "wav"
        elif arm == "vllmomni" and self.key == "omnivoice":  # diffusion speech path: one WAV, not streamed
            return {"path": "/v1/audio/speech", "format": "wav", "sample_rate": t["sr"],
                    "body": {"model": t["hf"], "input": "{text}", "response_format": "wav", "seed": "{seed}",
                             "extra_params": {"num_inference_steps": 32, "guidance_scale": 2.0}}}
        elif arm == "orpheusvllm":
            body = {"model": t["hf"], "input": "{text}", "stream": True, "response_format": "pcm", "seed": "{seed}"}
            fmt = "pcm"
        elif arm == "kokorofastapi":
            body = {"model": "kokoro", "input": "{text}", "speed": 1.0, "response_format": "pcm", "stream": True}
            fmt = "pcm"
        elif arm == "cbserver":  # its OpenAI route: no streaming, no pcm, needs a voice file
            return {"path": "/v1/audio/speech", "format": "wav", "sample_rate": t["sr"],
                    "body": {"model": "chatterbox", "input": "{text}", "voice": "Abigail.wav", "response_format": "wav",
                             "seed": "{seed}"}}
        elif arm == "vllmomni":
            body = {"model": t["hf"], "input": "{text}", "stream": True, "stream_format": "audio",
                    "response_format": "pcm", "seed": "{seed}"}
            fmt = "pcm"
        elif arm == "sgomni":
            body = {"model": t["hf"], "input": "{text}", "stream": True, "response_format": "pcm", "seed": "{seed}",
                    "voice": t["voice"] or "default", **t.get("sampling", {})}
            fmt = "pcm"
        elif arm == "voxserve":
            return {"path": "/generate", "form": True, "body": {"text": "{text}", "streaming": "true"},
                    "format": "wav", "sample_rate": t["sr"]}
        elif arm == "fishspeech":
            s = t.get("sampling", {})
            return {"path": "/v1/tts", "format": "wav", "sample_rate": t["sr"],
                    "body": {"text": "{text}", "format": "wav", "streaming": True, "references": [], "seed": "{seed}",
                             "temperature": s.get("temperature", 0.8), "top_p": s.get("top_p", 0.8),
                             "repetition_penalty": s.get("repetition_penalty", 1.0), "max_new_tokens": 2048,
                             "chunk_length": 200, "normalize": True}}
        else:
            raise ValueError(arm)
        if t["voice"] and arm != "sgomni":
            body["voice"] = t["voice"]
        if t["language"]:
            body["language"] = t["language"]
        return {"path": "/v1/audio/speech", "body": body, "format": fmt, "sample_rate": t["sr"]}

    def tclient(self, arm, out, conc, num, warm, seed_base=1000, offset=0, save=None):
        cmd = [str(dbench.CLIENT_PY), "-I", str(HERE / "dtts_client.py"), "--engine", arm, "--port", str(PORT0),
               "--spec", json.dumps(self.spec(arm)), "--texts", str(TEXTS), "--conc", str(conc), "--num", str(num),
               "--warmup-waves", str(warm), "--seed-base", str(seed_base), "--offset", str(offset),
               "--timeout", "900", "--out", str(out)]
        if save:
            cmd += ["--save-dir", str(save)]
        Path(out).mkdir(parents=True, exist_ok=True)
        rf = Path(out) / "results.json"
        rf.unlink(missing_ok=True)
        limit = int(600 + 60 * math.ceil((num + warm * conc) / conc) * (1 + conc / 8))
        return self.watched_client(cmd, out, limit, idle_s=900 if warm == 0 else 300)

    def concs_for(self, arm):
        if arm in OFFLINE:
            return OFFLINE[arm]["batches"]
        if arm == "cbserver":  # one request at a time by construction
            return (1, 2, 4, 8)
        return (1, 2, 4, 8) if arm == "fishspeech" else CONCS

    def knobs_of(self, arm, name=None):
        table = knob_table(self.key, arm)
        if name is None:
            name = self.tuned().get(arm, {}).get("config") or \
                (self.tuned().get("mstar_frontend", {}).get("config") if arm in TREES else None) or next(iter(table))
        return name, table[name]

    # ------------------------------------------------------------ passes
    def tts_pass(self, arm, rnd, tag, knob=None, points=None, gate_tag=None):
        kname, kargs = self.knobs_of(arm, knob)
        points = points or [(c, NUMS[c]) for c in self.concs_for(arm)]
        gate_done = gate_tag is None or (self.out / "gate" / f"{gate_tag}_cell" / "results.json").exists()
        if arm in OFFLINE:
            return self.offline_pass(arm, kname, kargs, rnd, tag, points, gate_tag, gate_done)
        last_err, last_log, last_cell = "", None, "server start"
        for attempt in (1, 2):
            done = self.done_cells(tag)
            todo = [(c, n) for c, n in points if (arm, kname, c, rnd, None) not in done]
            if not todo and gate_done:
                return True
            STATUS.update(phase=tag, arm=f"{arm}[{kname}] r{rnd} attempt {attempt}", cell="server start")
            reap_all("before tts start")
            lab = f"{tag}_{arm}_{kname}_r{rnd}" + ("" if attempt == 1 else "_retry")
            last_log = self.out / "servers" / f"{lab}.log"
            name = self.start(arm, lab, 0, PORT0, kargs)
            if not name:
                last_err = "server did not become healthy"
                continue
            try:
                s = self.tclient(arm, self.out / "servers" / f"smoke_{lab}", 1, 1, 0, seed_base=7)
                if s.get("failed") != 0:
                    raise RuntimeError(f"smoke failed: {s}")
                if points:  # untimed pass: graph buckets / compiles for every concurrency used later
                    STATUS.update(cell="warm pass")
                    self.tclient(arm, self.out / "cells" / tag / arm / f"warmpass_r{rnd}_{kname}",
                                 min(16, max(c for c, _ in points)), 64, 0, seed_base=900000)
                ok = True
                for c, n in todo:
                    last_cell = f"c={c}"
                    STATUS.update(cell=f"{tag} {arm}[{kname}] c={c} r{rnd}")
                    la0, t0 = loadavg(), time.time()
                    d = self.out / "cells" / tag / arm / f"c{c}_r{rnd}_{kname}"
                    s = self.tclient(arm, d, c, n, 1, seed_base=1000 + 10000 * rnd)
                    row = {"tag": tag, "arm": arm, "knob": kname, "conc": c, "round": rnd, "steps": None, "num": n,
                           "load_before": la0, "load_after": loadavg(), "started": t0,
                           "seconds": round(time.time() - t0, 1), **s}
                    self.append_row(row)
                    log(f"   {tag} {arm}[{kname}] c={c} r{rnd}: {s.get('audio_s_per_s') and round(s['audio_s_per_s'], 2)} "
                        f"audio-s/s rtf50 {s.get('rtf_p50') and round(s['rtf_p50'], 3)} ttfa50 "
                        f"{s.get('ttfa_p50') and round(s['ttfa_p50'], 3)} failed={s.get('failed')}")
                    if s.get("failed") != 0:
                        ok = False
                        last_err = f"cell failed={s.get('failed')} {s.get('errors') or s.get('error')}"
                        break
                if ok and not gate_done:
                    g = self.out / "gate"
                    STATUS.update(cell=f"gate collect {gate_tag}")
                    s = self.tclient(arm, g / f"{gate_tag}_cell", 8, GATE_N, 0, seed_base=1000, save=g / gate_tag)
                    fp = g / "failed.json"
                    failed = json.loads(fp.read_text()) if fp.exists() else {}
                    failed[gate_tag] = s.get("failed")
                    fp.write_text(json.dumps(failed))
                    gate_done = True
                if ok:
                    (self.out / "versions").mkdir(exist_ok=True)
                    (self.out / "versions" / f"{arm}.json").write_text(json.dumps(self.tversions(arm)))
                    return True
            except Exception as e:  # noqa: BLE001
                last_err = f"pass exception: {e!r}"
                log(f"!! {tag} {arm}: {e!r}")
            finally:
                self.stop(name)
                reap_all("after tts pass")
            if "illegal memory access" in last_err + str(getattr(self, "last_abort", "")):
                self.mark_knob_failed(arm, kname, "illegal memory access")
                break  # a config that corrupts GPU memory is not retried
            if attempt == 1:
                self.decision("retry", tag=tag, arm=arm, knob=kname, round=rnd)
        self.decision("pass_failed", tag=tag, arm=arm, knob=kname, round=rnd)
        failure(self.key, f"{arm}[{kname}]", tag, f"r{rnd} {last_cell}",
                f"{last_err}\n--- server log tail ---\n{tail(last_log) if last_log else ''}",
                "recorded as failed after one retry on a fresh server; queue moves on")
        return False

    def offline_run(self, arm, out, batches, nums, warm, seed_base, kargs, save=None):
        o = OFFLINE[arm]
        cmd = [o["python"], "-I", str(HERE / o["script"]), "--workload", "tts", "--items", str(TEXTS_JSONL),
               "--batch-sizes", ",".join(map(str, batches)), "--nums", ",".join(map(str, nums)), "--warmup", str(warm),
               "--seed-base", str(seed_base), "--out", str(out)] + list(kargs)
        if save:
            cmd += ["--save-dir", str(save)]
        env = {**os.environ, "HF_HOME": str(WORK / "hf"), "CUDA_VISIBLE_DEVICES": "0", **o.get("env", {})}
        tok = WORK / "hf/token"
        if tok.exists():
            env["HF_TOKEN"] = tok.read_text().strip()
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "results.json").unlink(missing_ok=True)
        lp = Path(str(out) + ".log")
        log(f"offline {arm} b={batches}: {shlex.join(cmd)}")
        limit = int(1800 + sum(60 * (math.ceil(n / b) + warm) for b, n in zip(batches, nums)))
        with open(lp, "w") as f:
            rc, reason = dbench.run_watched(cmd, limit, {"log": str(lp), "idle_s": 1200}, env=env, cwd=o["cwd"],
                                            stdout=f, stderr=subprocess.STDOUT)
        try:
            return json.loads((Path(out) / "results.json").read_text())["rows"], lp
        except (OSError, ValueError, KeyError):
            return [{"batch": b, "num": n, "failed": None, "error": f"rc={rc} {reason or ''} {tail(lp, 5)}"[:1500]}
                    for b, n in zip(batches, nums)], lp

    def offline_pass(self, arm, kname, kargs, rnd, tag, points, gate_tag, gate_done):
        """One offline process per attempt for the batch sizes not yet done, then (if asked) the gate WAVs
        (batch 8, GATE_N sentences, saved as <i>.wav like the served arms')."""
        last, lp = "", None
        for attempt in (1, 2):
            reap_all("before offline pass")
            done = self.done_cells(tag)
            todo = [(c, n) for c, n in points if (arm, kname, c, rnd, None) not in done]
            if not todo and gate_done:
                return True
            STATUS.update(phase=tag, arm=f"{arm}[{kname}] r{rnd} attempt {attempt}", cell=f"offline b={[c for c, _ in todo]}")
            ok = True
            if todo:
                la0, t0 = loadavg(), time.time()
                d = self.out / "cells" / tag / arm / f"r{rnd}_{kname}" / f"attempt{attempt}"
                rows, lp = self.offline_run(arm, d, [c for c, _ in todo], [n for _, n in todo], 1, 1000 + 10000 * rnd, kargs)
                for r in rows:
                    row = {"tag": tag, "arm": arm, "knob": kname, "conc": r["batch"], "round": rnd, "steps": None,
                           "num": r.get("num"), "load_before": la0, "load_after": loadavg(), "started": t0,
                           "seconds": round(time.time() - t0, 1), "offline_batch": True,
                           **{k: v for k, v in r.items() if k != "batch"}}
                    self.append_row(row)
                    log(f"   {tag} {arm}[{kname}] b={r['batch']} r{rnd} (offline): {r.get('audio_s_per_s')} audio-s/s "
                        f"failed={r.get('failed')}")
                    if r.get("failed") != 0:
                        ok, last = False, f"b={r['batch']} failed={r.get('failed')} {r.get('error') or r.get('errors')}"
            if ok and not gate_done:
                g = self.out / "gate"
                rows, lp = self.offline_run(arm, g / f"{gate_tag}_cell", [8], [GATE_N], 0, 1000, kargs, save=g / gate_tag)
                fp = g / "failed.json"
                failed = json.loads(fp.read_text()) if fp.exists() else {}
                failed[gate_tag] = rows[0].get("failed") if rows else None
                fp.write_text(json.dumps(failed))
                if rows and rows[0].get("failed") == 0:
                    gate_done = True
                else:
                    ok, last = False, f"gate collection failed: {rows[0] if rows else 'no rows'}"
            reap_all("after offline pass")
            if ok:
                (self.out / "versions").mkdir(exist_ok=True)
                (self.out / "versions" / f"{arm}.json").write_text(json.dumps(self.tversions(arm)))
                return True
            if attempt == 1:
                self.decision("retry", tag=tag, arm=arm, knob=kname, round=rnd)
        self.decision("pass_failed", tag=tag, arm=arm, knob=kname, round=rnd)
        failure(self.key, f"{arm}[{kname}]", tag, f"r{rnd}", f"{last}\n--- log tail ---\n{tail(lp) if lp else ''}",
                "recorded as failed after one retry on a fresh process; queue moves on")
        return False

    def tversions(self, arm):
        if arm in TREES:
            return self.versions(arm)
        if arm in ("fishspeech", "cbserver", "cbvllm", "kokorofastapi"):
            fs = ENG / {"fishspeech": "fish-speech", "cbserver": "cbserver", "cbvllm": "chatterbox-vllm",
                        "kokorofastapi": "kokoro-fastapi"}[arm]
            r = subprocess.run(["git", "-C", str(fs), "describe", "--tags", "--always"], capture_output=True, text=True)
            return {"git": r.stdout.strip()}
        if arm == "ovofficial":
            return {"omnivoice": "git 08be0b4 (overlay)", "python": OFFLINE[arm]["python"]}
        venv = {"vllmomni": "vllmomni", "sgomni": "sgomni", "voxserve": "voxserve", "orpheusvllm": "orpheusvllm"}[arm]
        r = subprocess.run(["uv", "pip", "freeze", "--python", str(ENG / venv / "bin/python")], capture_output=True,
                           text=True, env={**os.environ, "PATH": f"/usr/local/bin:{os.environ['PATH']}"})
        keep = ("vllm", "sglang", "torch=", "flashinfer", "transformers", "vox", "qwen", "triton", "snac")
        return {"packages": [x for x in r.stdout.splitlines() if x.lower().startswith(keep)]}

    def arms(self):
        dropped = json.loads((self.out / "dropped.json").read_text()) if (self.out / "dropped.json").exists() else {}
        return ["mstar-pr", "mstar-main"] + [c for c in self.t["comps"] if c not in dropped]


# ---------------------------------------------------------------- setup
SETUP_SH = r"""
set -uo pipefail
export PATH=/usr/local/bin:$HOME/.local/bin:$PATH HF_HOME=/root/work/hf
E=/root/work/engines; cd $E
case "$1" in
sgomni)
  ( [ -x sgomni/bin/sgl-omni ] && sgomni/bin/python -c "import sglang_omni" ) || {
    uv venv --python 3.12 --managed-python --seed sgomni &&
    uv pip install --python sgomni/bin/python --prerelease=allow "sglang-omni==0.1.9" &&
    (command -v sox >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq sox)) &&
    uv pip install --python sgomni/bin/python --no-deps sox einops "qwen-tts==0.1.1" &&
    uv pip install --python sgomni/bin/python "descript-audiotools==0.7.2" "descript-audio-codec==1.0.0"; } ;;
voxserve)
  ( [ -x voxserve/bin/vox-serve ] ) || {
    uv venv --python 3.12 --managed-python --seed voxserve &&
    uv pip install --python voxserve/bin/python "vox-serve==0.1.0"; } ;;
fishspeech)
  ( [ -x fish-speech/.venv/bin/python ] && fish-speech/.venv/bin/python -c "import fish_speech" ) || {
    rm -rf fish-speech && git clone -q -b v3.0.0-beta https://github.com/fishaudio/fish-speech &&
    cd fish-speech && uv sync --python 3.12 --extra cu129 && cd ..; }
  mkdir -p fish-speech/checkpoints
  [ -e fish-speech/checkpoints/s2-pro ] || ln -s "$(/root/work/mstar/.venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('fishaudio/s2-pro'))")" fish-speech/checkpoints/s2-pro ;;
orpheusvllm)
  ( [ -x orpheusvllm/bin/python ] && orpheusvllm/bin/python -c "import vllm, snac" ) || {
    uv venv --python 3.12 --managed-python --seed orpheusvllm &&
    uv pip install --python orpheusvllm/bin/python --torch-backend=auto "vllm==0.30.0" "snac==1.2.1"; } ;;
kokorofastapi)
  ( [ -x kokoro-fastapi/.venv/bin/uvicorn ] && [ -s kokoro-fastapi/api/src/models/v1_0/kokoro-v1_0.pth ] ) || {
    rm -rf kokoro-fastapi && git clone -q https://github.com/remsky/Kokoro-FastAPI kokoro-fastapi &&
    cd kokoro-fastapi && git checkout -q v0.9.0 && uv venv --python 3.12 --managed-python .venv &&
    VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[gpu]" &&
    .venv/bin/python docker/scripts/download_model.py --output api/src/models/v1_0 && cd ..; } ;;
cbserver)
  ( [ -x cbserver/.venv/bin/python ] && [ -s cbserver/.dbench_configured ] ) || {
    rm -rf cbserver && git clone -q https://github.com/devnen/Chatterbox-TTS-Server cbserver && cd cbserver &&
    git checkout -q 915ae28 && uv venv --python 3.10 --managed-python .venv &&
    VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements-nvidia.txt &&
    VIRTUAL_ENV=$PWD/.venv uv pip install --no-deps "git+https://github.com/devnen/chatterbox-v2.git@master" \
      "s3tokenizer==0.3.0" "onnx==1.16.0" &&
    .venv/bin/python - <<'PY' && echo ok > .dbench_configured && cd ..
import yaml
c = yaml.safe_load(open("config.yaml"))
c.setdefault("server", {}).update(host="127.0.0.1", port=8400)
c.setdefault("model", {})["repo_id"] = "chatterbox"
c.setdefault("tts_engine", {})["device"] = "cuda"
c.setdefault("generation_defaults", {}).update(temperature=0.8, exaggeration=0.5, cfg_weight=0.5)
yaml.safe_dump(c, open("config.yaml", "w"), sort_keys=False)
print(c["server"], c["model"], c["generation_defaults"])
PY
  } ;;
cbvllm)
  ( [ -x chatterbox-vllm/.venv/bin/python ] && chatterbox-vllm/.venv/bin/python -c "import chatterbox_vllm" ) || {
    rm -rf chatterbox-vllm && git clone -q https://github.com/randombk/chatterbox-vllm && cd chatterbox-vllm &&
    git checkout -q 8630314597985f085806983a5319247d1e067e9b && uv venv --python 3.12 --managed-python .venv &&
    (uv sync --python .venv/bin/python || VIRTUAL_ENV=$PWD/.venv uv pip install -e .) &&
    .venv/bin/python -c "import chatterbox_vllm" && cd ..; } ;;
ovofficial)
  /root/work/mstar/.venv/bin/python -c "import sys; sys.path.insert(0, '/root/work/overlays/omnivoice'); import omnivoice" ;;
overlay)
  shift; d=$1; shift; mkdir -p "$d"
  uv pip install --python /root/work/mstar/.venv/bin/python --no-deps --target "$d" "$@" ;;
rustfe)
  for t in mstar mstar-main; do
    [ -x /root/work/$t/rust/server/target/release/mstar-server ] || (. $HOME/.cargo/env; cargo build -q --release --manifest-path /root/work/$t/rust/server/Cargo.toml)
  done ;;
esac
"""


def write_configs(b):
    CFGDIR.mkdir(exist_ok=True)
    t = b.t
    if "sgomni" in t["comps"]:
        (CFGDIR / f"sgomni_{b.key}.yaml").write_text(f"config_cls: {t['sg_cfg']}\nmodel_path: {t['hf']}\n")
    if b.key == "omnivoice":  # tuned vllm-omni deploy: fp16 (M*'s and the reference's dtype), batch up to 32
        import yaml
        d = yaml.safe_load((VOSP / "deploy/omnivoice.yaml").read_text())
        for st in d["stages"]:
            st.update(dtype="float16", max_num_seqs=32)
        (CFGDIR / "vllmomni_omnivoice_fp16_mns32.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    if b.key == "fish_s2pro":
        import re
        for src, dst in (("fish_qwen3_omni_high_concurrency_single_gpu.yaml", "vllmomni_fish_hc_topp08.yaml"),
                         ("fish_qwen3_omni.yaml", "vllmomni_fish_default_topp08.yaml")):
            p = VOSP / "deploy" / src
            if p.exists():
                txt = re.sub(r"(top_p:\s*)[0-9.]+", r"\g<1>0.8", p.read_text())
                (CFGDIR / dst).write_text(txt)


def setup(b):
    env = {**os.environ, "HF_HOME": str(WORK / "hf"), "PATH": f"/usr/local/bin:{os.environ['PATH']}"}
    tok = WORK / "hf/token"
    if tok.exists():
        env["HF_TOKEN"] = tok.read_text().strip()
    # mstar trees, the client env and the diffusers env (Whisper judge) come from install_engines.sh
    subprocess.run(["bash", str(HERE / "install_engines.sh")], env=env, timeout=7200,
                   stdout=open(b.out / "setup.log", "a"), stderr=subprocess.STDOUT)
    r = subprocess.run([str(TREES["mstar-pr"] / ".venv/bin/python"), "-c",
                        f"from huggingface_hub import snapshot_download as s; s({b.t['hf']!r}); "
                        "s('openai/whisper-large-v3'); from huggingface_hub import hf_hub_download as h; "
                        "h('hexgrad/Kokoro-82M', 'samples/af_heart_0.wav')"],
                       env=env, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0 or not TEXTS.exists():
        raise RuntimeError(f"tts setup: weights/texts missing: {r.stderr[-400:]}")
    if not TEXTS_JSONL.exists():
        TEXTS_JSONL.write_text("".join(json.dumps({"text": x, "units": 1}) + "\n"
                                       for x in TEXTS.read_text().splitlines() if x.strip()))
    if b.t.get("overlay"):  # the M* arms' extra pure-Python packages, outside the shared venvs
        p = subprocess.run(["bash", "-c", SETUP_SH, "setup", "overlay", str(OVERLAY / b.key)] + b.t["overlay"], env=env,
                           timeout=3600, capture_output=True, text=True)
        (b.out / "setup_overlay.log").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            failure(b.key, "mstar", "setup", "overlay", (p.stdout + p.stderr)[-3000:], "M* arms will likely fail to start")
    steps = [c for c in b.t["comps"]] + (["rustfe"] if b.t["frontend"] else [])
    dropped = json.loads((b.out / "dropped.json").read_text()) if (b.out / "dropped.json").exists() else {}
    for s in steps:
        eng = s
        if s == "vllmomni":
            continue  # installed by install_engines.sh
        p = subprocess.run(["bash", "-c", SETUP_SH, "setup", s], env=env, timeout=7200, capture_output=True, text=True)
        (b.out / f"setup_{s}.log").write_text(p.stdout + p.stderr)
        if p.returncode != 0 and s != "rustfe":
            dropped[eng] = f"install failed rc={p.returncode}"
            failure(b.key, eng, "setup", "install", (p.stdout + p.stderr)[-3000:], "engine dropped from this model")
        elif p.returncode != 0:
            failure(b.key, "mstar", "setup", "rust frontend build", (p.stdout + p.stderr)[-3000:],
                    "frontend tune limited to python")
    write_configs(b)
    # per-engine smoke on GPU 0 alone: an engine that cannot serve one request is dropped
    for eng in [c for c in b.t["comps"] if c not in dropped] + ["mstar-pr"]:
        STATUS.update(phase="setup smoke", arm=eng)
        kname, kargs = b.knobs_of(eng, next(iter(knob_table(b.key, eng))))
        reap_all("before smoke")
        if eng in OFFLINE:
            rows, lp = b.offline_run(eng, b.out / "servers" / f"setup_smoke_{eng}", [1], [2], 0, 7, kargs)
            reap_all("after smoke")
            if not (rows and rows[0].get("failed") == 0):
                dropped[eng] = "smoke failed"
                failure(b.key, eng, "setup", "smoke", tail(lp, 30), "engine dropped from this model (setup smoke failed)")
            continue
        name = b.start(eng, f"setup_smoke_{eng}", 0, PORT0, kargs)
        ok = False
        if name:
            try:
                s = b.tclient(eng, b.out / "servers" / f"setup_smoke_{eng}_client", 1, 2, 0, seed_base=7)
                ok = s.get("failed") == 0
                log(f"setup smoke {eng}: {s}")
            finally:
                b.stop(name)
                reap_all("after smoke")
        if not ok and eng != "mstar-pr":
            dropped[eng] = "smoke failed"
            failure(b.key, eng, "setup", "smoke", tail(b.out / "servers" / f"setup_smoke_{eng}.log", 30),
                    "engine dropped from this model (setup smoke failed)")
        elif not ok:
            failure(b.key, eng, "setup", "smoke", tail(b.out / "servers" / f"setup_smoke_{eng}.log", 30),
                    "M* PR smoke failed; the model's timed phases will record failures")
    (b.out / "dropped.json").write_text(json.dumps(dropped, indent=1))
    b.decision("tts_arms", arms=b.arms(), dropped=dropped)


# ---------------------------------------------------------------- gate
def gate(b):
    g = b.out / "gate"
    tags = sorted(d.name for d in g.iterdir() if d.is_dir() and not d.name.endswith("_cell"))
    cmd = [str(dbench.DIFF_PY), "-I", str(HERE / "dtts_wer.py"), "--root", str(g), "--texts", str(TEXTS),
           "--count", str(GATE_N), "--tags", ",".join(tags), "--failed-json", str(g / "failed.json"),
           "--out", str(g)]
    env = {**os.environ, "HF_HOME": str(WORK / "hf"), "CUDA_VISIBLE_DEVICES": "7"}
    with open(g / "gate_run.log", "w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    try:
        v = json.loads((g / "gate.json").read_text())["verdict"]
    except (OSError, ValueError, KeyError):
        v = {}
        failure(b.key, "-", "gate", "-", tail(g / "gate_run.log"), "gate produced no verdicts")
    b.decision("gate", verdicts={k: x.get("verdict") for k, x in v.items()})
    for k, x in v.items():
        if not str(x.get("verdict", "")).startswith("PASS"):
            failure(b.key, k, "gate", "-", json.dumps(x)[:2000],
                    "knob set ineligible for selection" if "__" in k else
                    "gate FAIL recorded; this arm's timed numbers are not reported as a win")


def tune(b):
    for eng in b.arms():
        if eng == "mstar-main":
            continue
        table = knob_table(b.key, eng)
        if eng in TREES and len(table) == 1:
            continue
        for k in table:
            if b.marker(f"tune_{eng}_{k}").exists():
                continue
            pts = [(c, n) for c, n in TUNE if c in b.concs_for(eng)] or [(1, 16), (8, 48)]
            if eng == "fishspeech":
                pts = [(1, 16), (4, 32)]
            b.tts_pass(eng, 0, f"tune_{k}", k, pts, gate_tag=f"{eng.replace('-', '_')}__{k}")
            b.marker(f"tune_{eng}_{k}").touch()
    res = {}
    for r in b.rows():
        if r.get("tag", "").startswith("tune_") and "conc" in r:
            res.setdefault(r["arm"], {}).setdefault(r["knob"], {})[r["conc"]] = r
    out = {}
    for eng, sets in res.items():
        for k, rs in sets.items():
            ok = rs and all(x.get("failed") == 0 and x.get("audio_s_per_s") for x in rs.values())
            out.setdefault(eng, {})[k] = {
                "score": math.exp(statistics.fmean(math.log(x["audio_s_per_s"]) for x in rs.values())) if ok else None,
                "per_conc": {c: x.get("audio_s_per_s") for c, x in rs.items()},
                "ttfa_p50_c1": rs.get(1, {}).get("ttfa_p50")}
    (b.out / "tune.json").write_text(json.dumps(out, indent=1))


def select(b):
    """Rule: per engine, the knob set with the best tune score (geometric-mean audio-s/s over the tune
    points) among those whose gate passed; M*'s frontend pick applies to both M* arms."""
    tune_res = json.loads((b.out / "tune.json").read_text()) if (b.out / "tune.json").exists() else {}
    try:
        verdict = json.loads((b.out / "gate" / "gate.json").read_text())["verdict"]
    except (OSError, ValueError, KeyError):
        verdict = {}
    tuned = b.tuned()
    for eng, sets in tune_res.items():
        bad = b.knob_failed()
        scored = sorted(((v["score"], k) for k, v in sets.items() if v.get("score") and f"{eng}__{k}" not in bad),
                        reverse=True)
        passing = [(s, k) for s, k in scored
                   if verdict.get(f"{eng.replace('-', '_')}__{k}", {}).get("verdict", "").startswith("PASS")]
        pick = (passing or scored or [(None, next(iter(knob_table(b.key, eng))))])[0]
        key = "mstar_frontend" if eng in TREES else eng
        tuned[key] = {"config": pick[1], "score": pick[0], "gate_pass": bool(passing), "candidates": scored}
        b.decision("tuned_config", engine=key, config=pick[1], gate_pass=bool(passing), candidates=scored,
                   rule="best tune score among gate-passing knob sets")
    (b.out / "tuned.json").write_text(json.dumps(tuned, indent=1))


def collect(b):
    """M* PR twice (two processes) and M* main, on the chosen frontend, gate WAVs only (untimed)."""
    for arm, tag in (("mstar-pr", "mstar_pr"), ("mstar-pr", "mstar_pr_b"), ("mstar-main", "mstar_main")):
        if (b.out / "gate" / f"{tag}_cell" / "results.json").exists():
            continue
        b.tts_pass(arm, 0, f"collect_{tag}", None, [], gate_tag=tag)


def rounds(b, n=3, first=1):
    arms = b.arms()
    for rnd in range(first, first + n):
        k = (rnd - 1) % len(arms)
        for arm in arms[k:] + arms[:k]:
            b.tts_pass(arm, rnd, "main")


def profile(b):
    rows = {}
    for r in b.rows():
        if r.get("tag") == "main" and r.get("failed") == 0 and r.get("audio_s_per_s"):
            rows.setdefault((r["arm"], r["conc"]), []).append(r["audio_s_per_s"])
    mu = {k: statistics.fmean(v) for k, v in rows.items()}
    ratios = {}
    for c in CONCS:
        comp = [mu[(a, c)] for a in b.arms() if a not in TREES and (a, c) in mu]
        if ("mstar-pr", c) in mu and comp:
            ratios[c] = mu[("mstar-pr", c)] / max(comp)
    pts = [min(ratios, key=ratios.get)] if ratios else [32]
    if ratios and pts[0] != 1 and ratios.get(1, 2) < 1:
        pts.append(1)
    b.decision("profile_points", points=pts, ratios=ratios,
               rule="lowest mstar-pr / best competitor audio-s/s, plus c=1 if it trails there")
    t = TREES["mstar-pr"]
    for c in pts:
        reap_all("before profile")
        lab = f"profile_c{c}"
        _, kargs = b.knobs_of("mstar-pr")
        name = b.start("mstar-pr", lab, 0, PORT0, kargs, env_extra={"MSTAR_PHASE_TIMING": "100"})
        if not name:
            failure(b.key, "mstar-pr", "profile", f"c={c}", tail(b.out / "servers" / f"{lab}.log"), "no profile")
            continue
        try:
            b.tclient("mstar-pr", b.out / "profile" / f"client_c{c}", c, NUMS[c], 2)
        finally:
            b.stop(name)
            reap_all("after profile")
        code = ("import sys; from benchmark.worker_phases.phases import parse_log, segment, render; "
                "print(render(segment(parse_log(open(sys.argv[1]).read()))))")
        r = subprocess.run([f"{t}/.venv/bin/python", "-c", code, str(b.out / "servers" / f"{lab}.log")], cwd=t,
                           capture_output=True, text=True, timeout=600)
        (b.out / "profile").mkdir(exist_ok=True)
        (b.out / "profile" / f"phases_c{c}.txt").write_text(r.stdout + r.stderr)


def summarize(b):
    latest = {}
    for r in b.rows():
        if r.get("tag") == "main" and "conc" in r:
            latest[(r["arm"], r["conc"], r["round"])] = r
    cells = {}
    for (arm, c, _), r in latest.items():
        cells.setdefault((arm, c), []).append(r)
    arms = b.arms()
    L = [f"# {b.key}: streaming TTS, tag main", "",
         "Arm configs: " + "; ".join(f"{label(a)} = {b.knobs_of(a)[0]}" for a in arms), ""]
    for metric, unit, fmtd in (("audio_s_per_s", "audio-s/s", 2), ("rtf_p50", "RTF p50", 3),
                               ("ttfa_p50", "TTFA p50 (s)", 3), ("ttfa_p99", "TTFA p99 (s)", 3)):
        L += [f"## {unit}: mean [min-max] over rounds (n rounds); FAILED if any round failed", "",
              "| arm | " + " | ".join(f"c={c}" for c in CONCS) + " |", "|---|" + "---:|" * len(CONCS)]
        for a in arms:
            cols = []
            for c in CONCS:
                rs = cells.get((a, c), [])
                if not rs:
                    cols.append("-")
                    continue
                if any(r.get("failed") != 0 for r in rs):
                    cols.append(f"FAILED (n={len(rs)})")
                    continue
                v = [r[metric] for r in rs if r.get(metric) is not None]
                cols.append(f"{statistics.fmean(v):.{fmtd}f} [{min(v):.{fmtd}f}-{max(v):.{fmtd}f}] n={len(v)}"
                            if v else "-")
            L.append(f"| {label(a)} | " + " | ".join(cols) + " |")
        L.append("")
    g = b.out / "gate" / "gate.md"
    if g.exists():
        L += ["## Gate (WER, Whisper large-v3)", "", g.read_text().strip(), ""]
    (b.out / "summary.md").write_text("\n".join(L) + "\n")
    notes = [f"notes for {b.key}", "dropped engines: " + json.dumps(
        json.loads((b.out / "dropped.json").read_text()) if (b.out / "dropped.json").exists() else {})]
    p = b.out / "startup.jsonl"
    if p.exists():
        st = {}
        for x in p.read_text().splitlines():
            r = json.loads(x)
            st.setdefault(r["arm"], []).append(r["seconds"])
        notes.append("startup s: " + "; ".join(f"{a}: median {statistics.median(v):.0f} n={len(v)}" for a, v in st.items()))
    for log_ in sorted((b.out / "servers").glob("main_mstar*.log")):
        txt = log_.read_text(errors="replace")
        notes.append(f"{log_.stem}: failed captures {txt.count('ailed to capture')}, graph runtime rust "
                     f"{'graph runtime: rust' in txt}, rust frontend {'mstar-server' in txt}")
    (b.out / "notes.txt").write_text("\n".join(notes) + "\n")
    subprocess.run([sys.executable, "-I", str(HERE / "report_tts.py")], timeout=600)


def run_phase(q, spec, phase):
    b = TTSBench(spec["key"])
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
        rounds(b, spec.get("rounds", q.get("rounds", 3)))
    elif phase == "rounds_extra":
        rounds(b, 2, first=4)
    elif phase == "profile":
        profile(b)
    elif phase in ("summarize", "resummarize"):
        summarize(b)
    else:
        raise ValueError(phase)
