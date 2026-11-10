"""Stress and parity harness for the voice agent, for the GPU run (configs/voice_agent.yaml).

    python benchmark/voice_agent/stress.py --out results/ [--server URL | --launch] \
        [--phases parity,paths,load,faults,reference]

Each phase writes ``<out>/<phase>.json`` ({"phase", "ok", "checks": [...], "seconds"}) and, after it, checks that the
server is healthy, no request is left running, and the KV pages are back to where they were before the phase.

With ``--launch`` the harness starts the servers itself, one per arm, with the config's knobs and KV pool changed for
the arm (`ARMS`), and kills each by its pid at the end. Without it, it talks to a server you started with
``--enable-metrics`` (the KV gauge is read from /metrics) and the phases that need another arm are skipped.

Phases
  parity   a request with tool rounds against one plain prefill of the same rendered conversation, greedy: the answer
           must be the same tokens. Per arm: the default and pre-planning on (the LLM speculates in both).
  paths    typed no tool; python once and in parallel; spoken in; speak out; two agents; a second spawn refused.
  load     mixed traffic at c=1/8/32/64, then a 20% client-disconnect storm.
  multiturn  three-turn conversations, typed, with and without a tool round: the prompt tokens each turn looked up in
           the prefix cache and the ones it already held (the server's counters, read around each turn), so the next
           turn's reuse is a number. Run alone, so the deltas are the turn's.
  faults   (small-KV arm) backpressure; a tool that hangs, crashes or prints a lot; the budget used up mid round.
  reference  each member against its own reference (reference.py has the criteria): the served Qwen3.5 tokens against
           HF transformers, the heard transcripts against HF Whisper, the speech against the `kokoro` package. The
           served outputs are collected with the server up (``reference_served.json``). With ``--launch`` the server is
           then stopped so the references get the GPU, and the comparison writes ``reference_compare.json``; without
           it, stop the server and run ``--phases reference_compare``, which compares an earlier run's
           ``reference_served.json`` and nothing else. Needs ``pip install kokoro scipy``.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference  # noqa: E402

try:
    import yaml
except ImportError:  # only --launch needs it
    yaml = None

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "voice_agent.yaml"
GREEDY = {"temperature": 0.0, "top_k": 1}
TOOL_CALL = re.compile(r"<tool_call>")
KV_USED = "mstar_kv_pages_used"
KV_CACHED = "mstar_kv_pages_cached"
QUERIED = "mstar_prefix_cache_query_tokens_total"
HELD = "mstar_prefix_cache_hit_tokens_total"
RUNNING = "mstar_conductor_requests"

# the server arms: name -> changes to the config (a dotted path into it -> value)
ARMS = {
    "default": {},
    # The LLM node speculates its next decode step in every arm (enable_async_scheduling, on by default; no knob
    # turns it off for a non-TP node). The config ships with pre-planning off for Whisper; this arm turns it on, so
    # the speculated step is also pre-planned on its own thread.
    "pre_plan": {"knobs.pre_plan": True, "knobs.step_threads": "threaded"},
    "small_kv": {"resources.kv_cache.max_num_pages": 256, "max_concurrent_requests": 16},
}


# -- talking to the server ----------------------------------------------------------------------------------------


def served(result: dict) -> bool:
    """A request is served when it ended 200 with no error, in a chunk or as the reply's text."""
    return result["status"] == 200 and not result["error"]


class Server:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        # the process group the harness started (--launch) and where to put what a hang leaves
        self.group: int | None = None
        self.out: Path | None = None
        self.hung = False

    def dump_stacks(self, why: str) -> None:
        """Every server process's Python stacks, by py-spy, into ``<out>/stacks_<why>/``: what a stalled worker is
        doing is in them, and a stall leaves no traceback."""
        if self.group is None or self.out is None:
            return
        listing = subprocess.run(["ps", "-eo", "pid=,pgid=,args="], capture_output=True, text=True, check=False).stdout
        folder = self.out / f"stacks_{why}"
        folder.mkdir(parents=True, exist_ok=True)
        for line in listing.splitlines():
            pid, pgid, *args = line.split(None, 2)
            if int(pgid) != self.group:
                continue
            try:
                dump = subprocess.run(
                    ["py-spy", "dump", "--pid", pid], capture_output=True, text=True, timeout=60, check=False)
                text = dump.stdout + dump.stderr
            except (OSError, subprocess.TimeoutExpired) as error:
                text = f"py-spy failed: {error}"
            (folder / f"{pid}.txt").write_text(f"{' '.join(args)}\n\n{text}")

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def scrape(self) -> str | None:
        try:
            return requests.get(f"{self.url}/metrics", timeout=5).text
        except requests.RequestException:
            return None

    @staticmethod
    def total(text: str, name: str, **labels) -> float | None:
        """The sum of the samples of ``name`` (those whose labels include ``labels``) in one scrape, or None."""
        total, found = 0.0, False
        for line in text.splitlines():
            if not line.startswith(name) or line.startswith("#"):
                continue
            head, _, value = line.rpartition(" ")
            if all(f'{k}="{v}"' in head for k, v in labels.items()):
                total, found = total + float(value), True
        return total if found else None

    def metric(self, name: str, **labels) -> float | None:
        """``total`` over a fresh scrape, or None without metrics. A counter has no sample before its first
        count: with metrics served, that is zero."""
        text = self.scrape()
        if text is None:
            return None
        found = self.total(text, name, **labels)
        return found if found is not None else 0.0 if "\nmstar_" in text else None

    def kv_held(self) -> float | None:
        """Pages held beyond the prefix cache's: used less the pages the cache names, from one scrape so the two
        come from the same report (the server reports once a second). With nothing running this is what leaked.
        Blind spot: a lease a failed request never released sits on a page the cache names, so it is subtracted
        too; only an eviction (a burst of distinct prompts that fills the pool) would show it."""
        text = self.scrape()
        used = None if text is None else self.total(text, KV_USED)
        return None if used is None else used - (self.total(text, KV_CACHED) or 0.0)

    def generate(self, text=None, wav: Path | None = None, history=None, speak=False, kwargs=None, timeout=150,
                 request_id=None, abort_after: float | None = None) -> dict:
        """One /generate call; the reply text, audio bytes, the chunks' signals, and the status."""
        model_kwargs = {"max_output_tokens": 256, "enable_thinking": False, **(kwargs or {})}
        if history:
            model_kwargs["history"] = history
        data = {"output_modalities": "text,audio" if speak else "text", "model_kwargs": json.dumps(model_kwargs)}
        if text:
            data["text"] = text
        if request_id:
            data["request_id"] = request_id
        files = [("files", (wav.name, wav.read_bytes(), "audio/wav"))] if wav else None
        out = {"text": "", "heard": "", "audio": 0, "pcm": bytearray(), "status": None, "error": None, "chunks": 0}
        started = time.monotonic()
        try:
            with requests.post(f"{self.url}/generate", data=data, files=files, stream=True, timeout=timeout) as r:
                out["status"] = r.status_code
                if r.status_code != 200:
                    out["error"] = r.text[:300]
                    return out
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    out["chunks"] += 1
                    payload = base64.b64decode(chunk["data"])
                    signal_name = (chunk.get("metadata") or {}).get("signal", "")
                    if chunk["modality"] == "audio":
                        out["audio"] += len(payload)
                        out["pcm"] += payload
                    elif chunk["modality"] == "text" and signal_name == "asr_text":
                        out["heard"] += payload.decode("utf-8", "replace")
                        out.setdefault("heard_ids", []).extend((chunk.get("metadata") or {}).get("token_ids", []))
                    elif chunk["modality"] == "text":
                        out["text"] += payload.decode("utf-8", "replace")
                        out.setdefault("token_ids", []).extend((chunk.get("metadata") or {}).get("token_ids", []))
                    elif chunk["modality"] == "error":
                        out["error"] = payload.decode("utf-8", "replace")
                    if abort_after is not None and time.monotonic() - started > abort_after:
                        out["aborted"] = True
                        return out   # closing the connection is the disconnect
        except requests.Timeout as error:
            self.hung = True
            self.dump_stacks(f"hang_{int(time.time())}")
            raise TimeoutError(f"a request outlived {timeout} s ({error})") from error
        except requests.RequestException as error:
            out["error"] = f"{type(error).__name__}: {error}"
        # a worker's failure reaches the client as the request's text, not as an error chunk
        if out["error"] is None and out["text"].startswith("Error in worker"):
            out["error"] = out["text"][:300]
        out["seconds"] = round(time.monotonic() - started, 2)
        return out


KOKORO_RATE = 24000   # the server's audio is headerless 16-bit mono PCM at Kokoro's rate


def pcm_wav(path: Path, pcm: bytes, rate: int = KOKORO_RATE) -> Path:
    """The server's own speech as a clip to send back in: real speech with a known transcript."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(pcm))
    return path


def silence_wav(path: Path, seconds: float = 1.0) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return path


# -- phases ---------------------------------------------------------------------------------------------------------


class Phase:
    def __init__(self, name: str, server: Server, out: Path):
        self.name, self.server, self.out = name, server, out
        self.checks: list[dict] = []

    def check(self, what: str, ok: bool, detail=None) -> bool:
        self.checks.append({"check": what, "ok": bool(ok), "detail": detail})
        print(f"  [{'ok' if ok else 'FAIL'}] {what}" + (f"  {detail}" if detail and not ok else ""))
        return bool(ok)

    def finish(self, started: float, baseline: dict) -> dict:
        """The after-phase checks: healthy, nothing running, KV back to baseline."""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            running = self.server.metric(RUNNING, state="running")
            waiting = self.server.metric(RUNNING, state="waiting")
            kv = self.server.kv_held()
            settled = (running in (None, 0.0)) and (waiting in (None, 0.0)) and (
                kv is None or baseline["kv"] is None or kv <= baseline["kv"])
            if settled:
                break
            time.sleep(1.0)
        self.check("server healthy", self.server.healthy())
        self.check("no request left running or waiting", running in (None, 0.0) and waiting in (None, 0.0),
                   {"running": running, "waiting": waiting})
        self.check("KV pages back to baseline", kv is None or baseline["kv"] is None or kv <= baseline["kv"],
                   {"used": kv, "baseline": baseline["kv"]})
        result = {"phase": self.name, "ok": all(c["ok"] for c in self.checks), "checks": self.checks,
                  "seconds": round(time.monotonic() - started, 1)}
        (self.out / f"{self.name}.json").write_text(json.dumps(result, indent=2))
        return result


def baseline(server: Server) -> dict:
    return {"kv": server.kv_held()}


def run_phase(name: str, server: Server, out: Path, body) -> dict:
    print(f"== {name}")
    phase, base, started = Phase(name, server, out), baseline(server), time.monotonic()
    try:
        body(phase)
    except Exception as error:  # noqa: BLE001  a harness failure is a result, not a crash
        phase.check("phase ran to the end", False, f"{type(error).__name__}: {error}")
    return phase.finish(started, base)


CONVERSATION = [{"role": "user", "content": "Use python to compute 17 * 23, and tell me the result."}]


def phase_parity(phase: Phase) -> None:
    """Tool rounds against one plain prefill of the same rendered conversation: the same greedy answer."""
    s = phase.server
    first = s.generate(CONVERSATION[0]["content"], kwargs={**GREEDY, "max_output_tokens": 400})
    phase.check("the tool request answers", served(first) and bool(first["text"]), first.get("error"))
    calls = TOOL_CALL.findall(first["text"])
    phase.check("it called the tool", bool(calls))
    # what the model wrote up to its last answer, as the conversation a plain request would be given: its tool
    # turns and their results, rendered by the template, so the same tokens are in the KV by either route
    before, _, answer = first["text"].rpartition("</tool_call>")
    if not before:
        return
    plain_history = [CONVERSATION[0], {"role": "assistant", "content": before + "</tool_call>"},
                     {"role": "tool", "content": "391"}]
    plain = s.generate(None, history=plain_history, kwargs={**GREEDY, "max_output_tokens": 400})
    phase.check("the plain request answers", served(plain), plain.get("error"))
    same = plain["text"].strip() == answer.strip().removeprefix("<think>\n\n</think>").strip()
    phase.check("the answer after the tool equals a plain prefill of the same conversation", same,
                {"with_rounds": answer[-200:], "plain": plain["text"][-200:]})


def phase_paths(phase: Phase) -> None:
    s, tmp = phase.server, Path(tempfile.mkdtemp())

    def answers(name, result, contains=None):
        ok = served(result) and bool(result["text"] or result["audio"])
        if contains is not None:
            ok = ok and contains in result["text"]
        phase.check(name, ok, result.get("error") or result["text"][-200:])
        return result

    answers("typed, no tool", s.generate("Say hello in five words."))
    answers("python once", s.generate("Use python to print 6 * 7.", kwargs={"max_output_tokens": 400}), "42")
    answers("python in parallel", s.generate(
        "Use python twice, in one turn: once to print 2 + 2 and once to print 3 * 3. Then tell me both.",
        kwargs={"max_output_tokens": 500}))
    spoke = answers("speak out", s.generate("Say 'good morning' out loud using the speak tool.", speak=True,
                                            kwargs={"max_output_tokens": 300}))
    phase.check("audio came back", spoke["audio"] > 0, spoke["audio"])
    # the server's own speech, sent back in: Whisper hears it, and the reply follows from what it heard
    if spoke["audio"]:
        spoken = answers("spoken in", s.generate(wav=pcm_wav(tmp / "q.wav", spoke["pcm"])))
        phase.check("the transcript is what was said", "morning" in spoken["heard"].lower(), spoken["heard"])
        phase.check("the transcript carries no control token", "<|" not in spoken["heard"], spoken["heard"])
    answers("silence in", s.generate(wav=silence_wav(tmp / "silence.wav", 1.5)))
    agents = answers("two agents", s.generate(
        "Use spawn_agent exactly once with two tasks: 'the sum of 1..10' and 'the sum of 1..20'. Report both sums.",
        kwargs={"max_output_tokens": 600}))
    phase.check("the spawn markup reached the client", "spawn_agent" in agents["text"])
    phase.check("the agents' answers came back into the reply", "55" in agents["text"] and "210" in agents["text"],
                agents["text"][-300:])
    again = s.generate(
        "Call spawn_agent again for the task 'x', after you have the agents' answers.", history=[
            {"role": "user", "content": "Use spawn_agent for the task 'say hi'."}],
        kwargs={"max_output_tokens": 600})
    phase.check("a second spawn does not start more agents", served(again) and bool(again["text"]), again.get("error"))


MIX = [
    lambda s, i: s.generate(f"Say a word starting with {chr(97 + i % 26)}.", kwargs={"max_output_tokens": 32}),
    lambda s, i: s.generate(f"Use python to print {i} * {i}.", kwargs={"max_output_tokens": 300}),
    lambda s, i: s.generate("Count to five.", speak=True, kwargs={"max_output_tokens": 128}),
]


def phase_load(phase: Phase) -> None:
    s = phase.server
    for c in (1, 8, 32, 64):
        n = max(c * 2, 4)
        with ThreadPoolExecutor(c) as pool:
            results = list(pool.map(lambda i: MIX[i % len(MIX)](s, i), range(n)))
        failed = [r for r in results if r["status"] != 200 or r["error"]]
        phase.check(f"c={c}: {n} mixed requests, none failed", not failed,
                    [(r["status"], r["error"]) for r in failed[:3]])
        times = sorted(r.get("seconds", 0) for r in results)
        phase.check(f"c={c}: latency recorded", True, {"p50": times[len(times) // 2], "max": times[-1]})
    # a disconnect storm: a fifth of the clients leave a moment after they start
    n = 100
    rng = random.Random(0)
    with ThreadPoolExecutor(32) as pool:
        results = list(pool.map(
            lambda i: MIX[i % len(MIX)](s, i) if rng.random() > 0.2 else
            s.generate(f"Write a long story about {i}.", kwargs={"max_output_tokens": 400}, abort_after=0.3),
            range(n)))
    kept = [r for r in results if not r.get("aborted")]
    phase.check("disconnect storm: the clients that stayed were served",
                all(served(r) for r in kept),
                [(r["status"], r["error"]) for r in kept if r["status"] != 200 or r["error"]][:3])


def phase_multiturn(phase: Phase) -> None:
    """Each turn of a conversation sent with the whole history before it: how much of its prompt the cache held."""
    s = phase.server
    if s.metric(QUERIED) is None:
        phase.check("the prefix-cache counters are served (--enable-metrics)", False)
        return
    scripts = {
        "plain": ["Name a colour.", "Name another one.", "Which two did you name?"],
        "tool_round": ["Use python to print 6 * 7.", "Now use python to print 7 * 8.", "What were the two results?"],
    }
    rows = []
    for name, turns in scripts.items():
        history: list[dict] = []
        for n, text in enumerate(turns, 1):
            time.sleep(2.0)   # the counters are reported once a second: read what the last turn added first
            queried, held = s.metric(QUERIED), s.metric(HELD)
            result = s.generate(
                text, history=history, kwargs={**GREEDY, "max_output_tokens": 300, "return_token_ids": True})
            time.sleep(2.0)
            phase.check(f"{name} turn {n} answers", served(result), result["error"])
            asked, kept = s.metric(QUERIED) - queried, s.metric(HELD) - held
            rows.append({"conversation": name, "turn": n, "prompt_tokens": asked, "cached_tokens": kept,
                         "reply_tokens": len(result.get("token_ids", []))})
            history += [{"role": "user", "content": text}, {"role": "assistant", "content": result["text"]}]
    (phase.out / "multiturn_hits.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(f"  {row['conversation']:10s} turn {row['turn']}: "
              f"{row['cached_tokens']:.0f} of {row['prompt_tokens']:.0f} prompt tokens cached")
    later = [r for r in rows if r["turn"] > 1]
    phase.check("a later turn finds part of its prompt cached", all(r["cached_tokens"] > 0 for r in later), later)


def phase_faults(phase: Phase) -> None:
    s = phase.server
    # backpressure: more long requests than the KV pool holds at once
    with ThreadPoolExecutor(48) as pool:
        results = list(pool.map(lambda i: s.generate(f"Tell me about the number {i}.", kwargs={
            "max_output_tokens": 600}), range(48)))
    phase.check("a pool smaller than the load: every request ends served or refused, none hangs",
                all(r["status"] in (200, 429, 503) for r in results), [r["status"] for r in results][:10])
    hang = s.generate("Use python to run: import time; time.sleep(3600)", kwargs={"max_output_tokens": 300},
                      timeout=180)
    phase.check("a tool that hangs is cut by its time limit, the reply goes on", served(hang), hang["error"])
    crash = s.generate("Use python to run: 1/0", kwargs={"max_output_tokens": 300})
    phase.check("a tool that crashes is an error the model reads", served(crash),
                crash["error"])
    huge = s.generate("Use python to print 'x' * 10000000.", kwargs={"max_output_tokens": 300})
    phase.check("a tool that prints a lot is cut to what is left of the budget", served(huge), huge["error"])
    tight = s.generate("Use python to print 1 and keep calling it ten more times.", kwargs={"max_output_tokens": 40})
    phase.check("the budget used up mid-round ends the request cleanly", served(tight), tight["error"])


def phase_reference(phase: Phase) -> None:
    """Collect what the references are compared with: the agent's replies, what it heard and what it said."""
    s, folder = phase.server, phase.out / "reference_wavs"
    folder.mkdir(exist_ok=True)
    (phase.out / "reference_served.json").unlink(missing_ok=True)   # an earlier run's is never compared
    kwargs = {**GREEDY, "max_output_tokens": reference.LLM_MAX_TOKENS, "return_token_ids": True}
    llm = []

    def ask(kind: str, index: int, prompt: str | None, history=None) -> dict:
        r = s.generate(prompt, history=history, kwargs=kwargs)
        phase.check(f"{kind} {index} served", served(r) and bool(r["text"]), r.get("error"))
        item = {"kind": kind, "index": index, "prompt": prompt, "text": r["text"],
                "token_ids": r.get("token_ids", []), "error": r.get("error")}
        if history:
            item["history"] = history
        llm.append(item)
        return item

    for i, prompt in enumerate(reference.PLAIN):
        ask("plain", i, prompt)
    calls = [ask("python", i, prompt) for i, (prompt, _) in enumerate(reference.PYTHON)]
    for i, ((prompt, result), call) in enumerate(zip(reference.PYTHON, calls, strict=True)):
        before, closed, _ = call["text"].partition(reference.TOOL_CLOSE)
        phase.check(f"multiturn {i}: the agent's own call is there to build the history from", bool(closed))
        history = [{"role": "user", "content": prompt}, {"role": "assistant", "content": before + closed},
                   {"role": "tool", "content": result}]
        ask("multiturn", i, None, history=history)

    asr = []
    for i, said in enumerate(reference.CLIP_SAY):
        spoke = s.generate(reference.say_prompt(said), speak=True, kwargs={**GREEDY, "max_output_tokens": 200})
        phase.check(f"clip {i} made", served(spoke) and spoke["audio"] > 0, spoke["error"])
        clip = pcm_wav(folder / f"clip_{i}.wav", spoke["pcm"])
        heard = s.generate(wav=clip, kwargs={**GREEDY, "max_output_tokens": 16, "return_token_ids": True})
        phase.check(f"clip {i} heard", served(heard), heard["error"])
        asr.append({"index": i, "said": said, "heard": heard["heard"], "heard_ids": heard.get("heard_ids", [])})

    tts = []
    for i, sentence in enumerate(reference.SENTENCES):
        spoke = s.generate(reference.say_prompt(sentence), speak=True, kwargs={**GREEDY, "max_output_tokens": 200})
        phase.check(f"sentence {i} spoken", served(spoke) and spoke["audio"] > 0, spoke["error"])
        pcm_wav(folder / f"tts_{i}_served.wav", spoke["pcm"])
        text, voice = reference.spoken_text(spoke["text"])
        tts.append({"index": i, "sentence": sentence, "spoken": text, "voice": voice})
    (phase.out / "reference_served.json").write_text(json.dumps({"llm": llm, "asr": asr, "tts": tts}, indent=2))


def compare_references(out: Path) -> dict:
    """The comparison, in this process once the servers are stopped (it loads the references on the GPU)."""
    print("== reference_compare")
    phase, started = Phase("reference_compare", None, out), time.monotonic()  # type: ignore[arg-type]
    try:
        reference.compare(out, phase.check)
    except Exception as error:  # noqa: BLE001  a harness failure is a result, not a crash
        phase.check("comparison ran to the end", False, f"{type(error).__name__}: {error}")
    result = {"phase": phase.name, "ok": all(c["ok"] for c in phase.checks), "checks": phase.checks,
              "seconds": round(time.monotonic() - started, 1)}
    (out / "reference_compare.json").write_text(json.dumps(result, indent=2))
    return result


PHASES = {"parity": (phase_parity, ("default", "pre_plan")), "paths": (phase_paths, ("default",)),
          "load": (phase_load, ("default",)), "multiturn": (phase_multiturn, ("default",)),
          "faults": (phase_faults, ("small_kv",)), "reference": (phase_reference, ("default",))}


# -- launching an arm -----------------------------------------------------------------------------------------------


def arm_config(arm: str, dest: Path) -> Path:
    if yaml is None:
        raise SystemExit("--launch needs pyyaml")
    config = yaml.safe_load(CONFIG.read_text())
    for dotted, value in ARMS[arm].items():
        *path, leaf = dotted.split(".")
        target = config
        for key in path:
            target = target.setdefault(key, {})
        target[leaf] = value
    path = dest / f"voice_agent_{arm}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def launch(arm: str, port: int, dest: Path) -> tuple[subprocess.Popen, Server]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    process = subprocess.Popen(
        [sys.executable, "-m", "mstar.cli.main", "serve", "voice_agent", "--config", str(arm_config(arm, dest)),
         "--port", str(port), "--enable-metrics"],
        env=env, stdout=(dest / f"server_{arm}.log").open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    server = Server(f"http://localhost:{port}")
    server.group, server.out = process.pid, dest
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"the {arm} server exited with {process.returncode}; see server_{arm}.log")
        if server.healthy():
            return process, server
        time.sleep(5)
    stop(process)
    raise SystemExit(f"the {arm} server did not come up")


def group_alive(pgid: int) -> bool:
    """Whether any process of the group is still running (a zombie holds no GPU, and an orphan one stays a zombie when
    PID 1 does not reap it)."""
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rpartition(")")[2].split()   # after the command name: state, ppid, pgrp, ...
        except OSError:
            continue   # it exited while we looked
        if int(fields[2]) == pgid and fields[0] != "Z":
            return True
    return False


def stop(process: subprocess.Popen) -> None:
    """End the server and its workers, by the process group the harness started (not by name)."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    # the workers hold the GPU until they are gone; whatever is left after the TERM is killed
    for _ in range(2):
        deadline = time.monotonic() + 60
        while group_alive(process.pid) and time.monotonic() < deadline:
            time.sleep(1)
        if not group_alive(process.pid):
            break
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()   # the leader is our child: reap it
    if group_alive(process.pid):
        raise SystemExit(f"the server's process group {process.pid} is still running after SIGKILL")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("stress_results"))
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--launch", action="store_true", help="start a server per arm")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--phases", default=",".join(PHASES))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    asked = [p for p in args.phases.split(",") if p]
    wanted = [p for p in asked if p != "reference_compare"]
    results = []
    if "reference" in wanted or "reference_compare" in asked:
        reference.preflight()
    arms = {arm for p in wanted for arm in PHASES[p][1]}
    for arm in sorted(arms, key=list(ARMS).index):
        process = None
        server = Server(args.server)
        if args.launch:
            process, server = launch(arm, args.port, args.out)
        elif arm != "default":
            print(f"-- skipping the {arm} arm: it needs --launch")
            continue
        try:
            for name in wanted:
                body, phase_arms = PHASES[name]
                if arm in phase_arms and not server.hung:
                    result = run_phase(name if len(phase_arms) == 1 else f"{name}_{arm}", server, args.out, body)
                    results.append(result)
        finally:
            if process is not None:
                stop(process)
    if "reference_compare" in asked or (args.launch and any(r["phase"] == "reference" for r in results)):
        results.append(compare_references(args.out))
    elif any(r["phase"] == "reference" for r in results):
        print("-- the references need the GPU: stop the server, then run --phases reference_compare")
    (args.out / "summary.json").write_text(json.dumps(
        [{"phase": r["phase"], "ok": r["ok"], "seconds": r["seconds"]} for r in results], indent=2))
    failed = [r["phase"] for r in results if not r["ok"]]
    print("FAILED: " + ", ".join(failed) if failed else "all phases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
