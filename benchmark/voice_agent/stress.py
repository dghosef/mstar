"""Stress and parity harness for the voice agent, for the GPU run (configs/voice_agent.yaml).

    python benchmark/voice_agent/stress.py --out results/ [--server URL | --launch] [--phases parity,paths,load,faults]

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

The KV-management phases (`mm/3-kv-management`; each arm is a server, `ARMS`):
  flashstate   (default and messages arms: the recurrent pool's `checkpoints: rows` and `checkpoints: messages`) the
           same long-history conversations, typed and with tool rounds, 5 turns: per turn the prompt tokens, the
           cached ones and the time to the first chunk. `flashstate_rows_<arm>.json` holds the rows (a phase's own
           `<name>.json` is its checks).
  diverge      (default and messages arms) three requests that share a long system block, tools and a first exchange
           and differ in their last message: A, then B, then C. Linear conversations extend the last prompt, so
           the row-end snapshot already covers them; only a request that diverges inside a shared prefix needs a
           snapshot at the boundary. B's cached tokens must reach the shared prefix on `messages`; `rows` is recorded.
  subagents    a parent that spawns 2 and 4 children: the cached tokens the children's first prefills found, from the
           counters' deltas against a control request of the same shape that spawns nothing.
  eviction     (pressure arm: a small KV pool) many sessions cycling long-history turns: the hit rate, preemptions and
           drops, and the pages the cache names, sampled through the run. The counters that exist; there is none
           for the recurrent pool's snapshots, so orphaned snapshots are not counted.
  rebuild_reference  (default arm) the long tool-round and short spoken requests, all at once on an unconstrained
           server, twice (the run-to-run variance), and the pool size the next phase needs. Run before `rebuild`.
  rebuild      (preempt arm: a pool the long answers outgrow) the same requests: the arm must have dropped requests,
           and every answer must be whole. Token-exactness is information (per-request first difference against
           the reference and its rerun), required only of the short spoken requests; rebuild exactness is pinned
           on the CPU (`test_worker_rebuild`).
A run of just these: --phases flashstate,diverge,subagents,eviction,rebuild_reference,rebuild
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

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
KV_TOTAL = "mstar_kv_pages_total"
PREEMPTIONS = "mstar_kv_preemptions_total"
DROPS = "mstar_kv_drops_total"
REBUILD_TOKENS = "mstar_kv_rebuild_tokens_total"
REBUILD_CACHED = "mstar_kv_rebuild_cached_pages_total"
QUERIED = "mstar_prefix_cache_query_tokens_total"
HELD = "mstar_prefix_cache_hit_tokens_total"
RUNNING = "mstar_conductor_requests"

# pages: below this one long request's footprint may not fit at all (the `preempt` arm's floor)
PREEMPT_FLOOR = 48

# the server arms: name -> changes to the config (a dotted path into it -> value)
ARMS = {
    "default": {},
    # The LLM node speculates its next decode step in every arm (enable_async_scheduling, on by default; no knob
    # turns it off for a non-TP node). The config ships with pre-planning off for Whisper; this arm turns it on, so
    # the speculated step is also pre-planned on its own thread.
    "pre_plan": {"knobs.pre_plan": True, "knobs.step_threads": "threaded"},
    "small_kv": {"resources.kv_cache.max_num_pages": 256, "max_concurrent_requests": 16},
    # where a prefill keeps recurrent snapshots: the pool's checkpoint policy (`rows` is the default)
    "messages": {"resources.gdn_state.checkpoints": "messages"},
    # pools sized in the LLM's 128-token pages (the spec logs `page_size=None`; the prefix counters move in 128s). The
    # eviction phase measures one session's pages and runs enough sessions to hold twice the pool; a tool request
    # reserves about 25 pages, so 64 holds a few of the 16 the rebuild phase sends at once
    "pressure": {"resources.kv_cache.max_num_pages": 96, "max_concurrent_requests": 64},
    # sized by `rebuild_reference` from the pages one long request takes (`PREEMPT_FLOOR` pages at least)
    "preempt": {"resources.kv_cache.max_num_pages": 64, "max_concurrent_requests": 16},
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
        self.log: Path | None = None
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

    def dropped_requests(self) -> set[str]:
        """The client request ids the server's log says the pool dropped (a launched server's)."""
        if self.log is None or not self.log.exists():
            return set()
        return set(re.findall(r"dropped (\S+?)#\d+", self.log.read_text(errors="replace")))

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
        out = {"text": "", "heard": "", "audio": 0, "pcm": bytearray(), "status": None, "error": None, "chunks": 0,
               "ttft": None}
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
                    if out["ttft"] is None and chunk["modality"] in ("text", "audio"):
                        out["ttft"] = round(time.monotonic() - started, 3)
                    signal_name = (chunk.get("metadata") or {}).get("signal", "")
                    if chunk["modality"] == "audio":
                        out["audio"] += len(payload)
                        out["pcm"] += payload
                    elif chunk["modality"] == "text" and signal_name == "asr_text":
                        out["heard"] += payload.decode("utf-8", "replace")
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
        frames = traceback.extract_tb(error.__traceback__)[-3:]
        phase.check("phase ran to the end", False, {
            "error": f"{type(error).__name__}: {error}",
            "frames": [f"{Path(f.filename).name}:{f.lineno} in {f.name}: {f.line}" for f in frames]})
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


# -- the KV-management phases ---------------------------------------------------------------------------------------

SYSTEM = ("You are a careful assistant for a long-running voice agent. " + "Answer briefly. " * 5 +
          "Background notes the user may ask about: " +
          " ".join(f"Note {i}: the {w} is kept on shelf {i} of the archive." for i, w in enumerate(
              ["atlas", "lantern", "compass", "ledger", "violin", "kettle", "barometer", "telescope"] * 6)))
LONG_SCRIPTS = {
    "plain": ["Name a colour.", "Name another one.", "Which two did you name?", "Name a third, different one.",
              "List all three in one line."],
    "tool_round": ["Use python to print 6 * 7.", "Now use python to print 7 * 8.", "Now print 8 * 9 with python.",
                   "And 9 * 10.", "What were the four results?"],
}


def counters(s: Server) -> dict:
    """The prefix-cache and preemption counters, from one scrape (the server reports once a second)."""
    text = s.scrape() or ""
    return {name: Server.total(text, name) or 0.0
            for name in (QUERIED, HELD, PREEMPTIONS, DROPS, REBUILD_TOKENS, REBUILD_CACHED)} | {
        # the LLM's pool alone: the gauges also report Whisper's two caches, which the unlabelled sum would add in
        "kv_cached": Server.total(text, KV_CACHED, resource="kv_cache") or 0.0,
        "kv_total": Server.total(text, KV_TOTAL, resource="kv_cache") or 0.0,
        "kv_used": Server.total(text, KV_USED, resource="kv_cache") or 0.0}


def settled(s: Server) -> dict:
    """`counters` once the server has reported what the last request did."""
    time.sleep(2.5)
    return counters(s)


def conversation(s: Server, turns: list[str], system: str | None = SYSTEM, measure: bool = True) -> list[dict]:
    """One conversation, each turn sent with the whole history before it: its prompt tokens, the cached ones, its
    time to the first chunk and its reply. ``measure`` reads the counters around each turn, which is only right for
    a conversation run alone; concurrent ones skip it."""
    history: list[dict] = [{"role": "system", "content": system}] if system else []
    rows = []
    for n, text in enumerate(turns, 1):
        before = settled(s) if measure else None
        result = s.generate(
            text, history=history, kwargs={**GREEDY, "max_output_tokens": 300, "return_token_ids": True})
        row = {"turn": n, "ttft": result["ttft"], "served": served(result), "error": result["error"],
               "reply_tokens": len(result.get("token_ids", [])), "token_ids": result.get("token_ids", []),
               "text": result["text"]}
        if measure:
            after = settled(s)
            row |= {"prompt_tokens": after[QUERIED] - before[QUERIED], "cached_tokens": after[HELD] - before[HELD]}
        rows.append(row)
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": result["text"]}]
    return rows


def phase_flashstate(phase: Phase) -> None:
    """Per turn: prompt tokens, cached tokens and time to the first chunk, for each conversation. Run once per
    checkpoint policy arm; the second arm's rows are set against the first's when its file is there."""
    s, arm = phase.server, phase.name.removeprefix("flashstate_")
    if s.metric(QUERIED) is None:
        phase.check("the prefix-cache counters are served (--enable-metrics)", False)
        return
    rows = []
    for name, turns in LONG_SCRIPTS.items():
        for row in conversation(s, turns):
            rows.append({"arm": arm, "conversation": name, **{k: v for k, v in row.items() if k != "token_ids"}})
            phase.check(f"{name} turn {row['turn']} answers", row["served"], row["error"])
    (phase.out / f"flashstate_rows_{arm}.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(f"  {arm:8s} {row['conversation']:10s} turn {row['turn']}: {row['cached_tokens']:.0f} of "
              f"{row['prompt_tokens']:.0f} prompt tokens cached, first chunk {row['ttft']} s")
    later = [r for r in rows if r["turn"] > 1]
    phase.check("a later turn finds part of its prompt cached", all(r["cached_tokens"] > 0 for r in later), later)
    other = phase.out / "flashstate_rows_default.json"
    if arm != "default" and other.exists():
        base = {(r["conversation"], r["turn"]): r for r in json.loads(other.read_text())}
        gain = [{"conversation": r["conversation"], "turn": r["turn"],
                 "cached_tokens_vs_rows": r["cached_tokens"] - base[(r["conversation"], r["turn"])]["cached_tokens"],
                 "ttft_vs_rows": None if None in (r["ttft"], base[(r["conversation"], r["turn"])]["ttft"])
                 else round(r["ttft"] - base[(r["conversation"], r["turn"])]["ttft"], 3)}
                for r in rows if (r["conversation"], r["turn"]) in base]
        phase.check("the arm's gain over `rows`, turn by turn (information)", True, gain)


def server_prompt_ids(history: list[dict], question: str) -> list[int]:
    """The prompt ids the server tokenizes for a typed turn: its own `process_prompt` on a model built without
    weights (the tokenizer from the HF cache), with the config's tools, so the template, the tools block and the
    rendering of the history are the server's and not a re-implementation of them."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    sys.path.insert(0, str(REPO))
    from mstar.model.multimodal import TEXT
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = object.__new__(Qwen3_5DenseModel)
    model.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
    model.media, model.vision_config, model._submodule_cache, model.sampling_defaults = (), None, {}, {}
    model.config = SimpleNamespace(max_position_embeddings=32768)
    model.tools = tuple(yaml.safe_load(CONFIG.read_text())["model_kwargs"]["members"]["llm"]["tools"])
    model.stream_in = ""
    out = model.process_prompt(question, [TEXT], ["text"], history=history, enable_thinking=False)
    return out.new_input_tensors["text_inputs"][0].tolist()


def common_prefix(a: list[int], b: list[int]) -> int:
    return next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


def diverge_question(i: int) -> str:
    """A question whose tail is about 700 tokens and unlike the others', so the three requests share their opening
    and nothing after it, and the shared prefix ends pages before each request's own last boundary."""
    words = ["atlas", "lantern", "compass", "ledger", "violin", "kettle", "barometer", "telescope"]
    tail = " ".join(f"Entry {j} of the {chr(65 + i)} list says the {words[(j + i) % len(words)]} number {j * (i + 2)}."
                    for j in range(60))
    return f"What is kept on shelf {3 + 4 * i}? First read this list, which you may ignore: {tail}"


DIVERGE_QUESTIONS = [diverge_question(i) for i in range(3)]


SHARED_TARGET = 1800   # tokens: its page floor, 1792, sits 384 above the first chunk end and 640 below the second
PAGE = 128
CHUNK_GUESSES = (1024, 2048)   # the prefill chunk the scheduler cuts A into (1024 in try 3); both are checked


def exchange_for(target: int) -> tuple[list[dict], int]:
    """The system block and a first exchange whose answer is padded until the A/B prompts share ``target`` tokens (as
    the server counts them), and that shared length."""
    system = f"tag {random.Random().randrange(10**9)}: {SYSTEM}"
    answer, sentence = "The atlas is kept on shelf 0 of the archive.", 0
    while True:
        exchange = [{"role": "system", "content": system}, {"role": "user", "content": "Which shelf holds the atlas?"},
                    {"role": "assistant", "content": answer}]
        shared = common_prefix(server_prompt_ids(exchange, DIVERGE_QUESTIONS[0]),
                               server_prompt_ids(exchange, DIVERGE_QUESTIONS[1]))
        if shared >= target:
            return exchange, shared
        answer += "".join(f" Shelf {sentence + k} holds notes of the archive." for k in range(10))
        sentence += 10


def rows_kept_points(restored: int, prompt: int, chunk: int) -> list[int]:
    """The boundaries the default policy leaves a snapshot at when A is prefilled from ``restored`` in chunks of
    ``chunk``: each chunk's end (a chunk with more to come keeps its own end), A's last boundary, and the
    restored one."""
    ends = [restored + k * chunk for k in range(1, (prompt - restored - 1) // chunk + 1)]
    return sorted({restored, *ends, (prompt - 1) // PAGE * PAGE})


def phase_diverge(phase: Phase) -> None:
    """A, B and C: the same system block, tools and first exchange, each ending in a different question. The system
    block opens with a tag no earlier request used, so the shared prefix is new to this server. B diverges inside
    what A left in the cache; C finds what B's prefill kept. The shared length is set so that its page floor is at
    least two pages from every boundary the default policy leaves a snapshot at while prefilling A (its chunk ends
    and its last boundary), so only a snapshot at the end of the first exchange (`messages`) can serve B."""
    s, arm = phase.server, phase.name.removeprefix("diverge_")
    if s.metric(QUERIED) is None:
        phase.check("the prefix-cache counters are served (--enable-metrics)", False)
        return
    exchange, shared = exchange_for(SHARED_TARGET)
    floor = shared // PAGE * PAGE
    rows = []
    for name, question in zip("ABC", DIVERGE_QUESTIONS, strict=True):
        before = settled(s)
        r = s.generate(question, history=exchange, kwargs={**GREEDY, "max_output_tokens": 24})
        after = settled(s)
        phase.check(f"{name} served", served(r), r["error"])
        rows.append({"arm": arm, "request": name, "prompt_tokens": after[QUERIED] - before[QUERIED],
                     "cached_tokens": after[HELD] - before[HELD], "ttft": r["ttft"], "shared_prefix_tokens": shared,
                     "shared_prefix_page_floor": floor})
    restored, prompt = int(rows[0]["cached_tokens"]), int(rows[0]["prompt_tokens"])
    points = {chunk: rows_kept_points(restored, prompt, chunk) for chunk in CHUNK_GUESSES}
    # what `rows` can serve B: the highest boundary A left a snapshot at that is within the shared prefix
    predicted_rows = {chunk: max(p for p in kept if p <= floor) for chunk, kept in points.items()}
    nearest = min(abs(floor - p) for kept in points.values() for p in kept)
    for row in rows:
        row["rows_kept_points_for_A"] = points
        row["predicted_B_cached_rows"] = predicted_rows
        row["predicted_B_cached_messages"] = floor
    (phase.out / f"diverge_rows_{arm}.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(f"  {arm:8s} {row['request']}: {row['cached_tokens']:.0f} of {row['prompt_tokens']:.0f} cached "
              f"(shared prefix {shared}, page floor {floor}), first chunk {row['ttft']} s")
    print(f"  A restored {restored} of {prompt}; rows keeps {points}; predicted B: rows {predicted_rows}, "
          f"messages {floor}")
    phase.check("the shared prefix's page floor is 2 pages from every boundary rows keeps for A", nearest >= 2 * PAGE,
                {"floor": floor, "nearest_distance": nearest, "points": points})
    b_cached = rows[1]["cached_tokens"]
    if arm == "messages":
        phase.check("B reaches the shared prefix (within a page)", b_cached >= floor - PAGE, rows)
        phase.check("B is served from beyond what `rows` could (predicted)", b_cached > max(predicted_rows.values()),
                    {"B": b_cached, "predicted_rows": predicted_rows})
        other = phase.out / "diverge_rows_default.json"
        if other.exists():
            rows_b = json.loads(other.read_text())[1]["cached_tokens"]
            phase.check("messages-B > rows-B (measured, the rows arm)", b_cached > rows_b,
                        {"messages": b_cached, "rows": rows_b})
    else:
        phase.check("rows: B and C recorded, set against the prediction (information)", True,
                    {"B": b_cached, "C": rows[2]["cached_tokens"], "predicted_B": predicted_rows})


def spawn_markup(text: str) -> int:
    """The agent tasks a reply's spawn call wrote (its markup reaches the client as text)."""
    return len(re.findall(r"<parameter=task", text)) or len(re.findall(r'"task"', text))


def phase_subagents(phase: Phase) -> None:
    """A parent that spawns n children: what the children's first prefills found cached. The counters are global, so
    it is a delta against a control of the same shape that spawns nothing (the parent's own prompt hits the same
    shared block in both, each prompt's own words being new)."""
    s = phase.server
    if s.metric(QUERIED) is None:
        phase.check("the prefix-cache counters are served (--enable-metrics)", False)
        return
    rows = []
    for n, tasks in ((2, ("the sum of 1..10", "the sum of 1..20")),
                     (4, ("the sum of 1..5", "the sum of 1..6", "the sum of 1..7", "the sum of 1..8"))):
        tag = f"run {random.Random(n).randrange(10**6)}"
        control = f"{tag}: use python to print {n} * {n}, once, and say the result."
        spawn = (f"{tag}: call spawn_agent {n} times in this turn, one call per task and one task in each call, "
                 f"for these tasks: " + "; ".join(f"'{t}'" for t in tasks) + ". Then report all the answers.")
        before = settled(s)
        c = s.generate(control, kwargs={**GREEDY, "max_output_tokens": 300})
        mid = settled(s)
        p = s.generate(spawn, kwargs={**GREEDY, "max_output_tokens": 600})
        after = settled(s)
        phase.check(f"{n} children asked: both requests served", served(c) and served(p), (c["error"], p["error"]))
        started = spawn_markup(p["text"])
        extra_held = (after[HELD] - mid[HELD]) - (mid[HELD] - before[HELD])
        extra_asked = (after[QUERIED] - mid[QUERIED]) - (mid[QUERIED] - before[QUERIED])
        row = {"children_asked": n, "children_started": started, "control_cached": mid[HELD] - before[HELD],
               "control_prompt": mid[QUERIED] - before[QUERIED], "spawn_cached": after[HELD] - mid[HELD],
               "spawn_prompt": after[QUERIED] - mid[QUERIED], "children_cached_total": extra_held,
               "children_prompt_total": extra_asked,
               "cached_per_child": round(extra_held / started, 1) if started else None}
        rows.append(row)
        phase.check(f"{n} asked, {started} started: enough children started", started >= (2 if n == 4 else 1),
                    p["text"][-200:])
        phase.check(f"{n} asked, {started} started: they found part of their prompt cached", extra_held > 0, row)
    (phase.out / "subagents_rows.json").write_text(json.dumps(rows, indent=2))


class Sampler:
    """The counters every ``every`` seconds on a thread of its own, between ``with`` and exit."""

    def __init__(self, s: Server, every: float = 1.0):
        self.s, self.every, self.series, self._stop = s, every, [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            self.series.append({"t": round(time.monotonic() - t0, 1), **counters(self.s)})
            self._stop.wait(self.every)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def phase_eviction(phase: Phase) -> None:
    """Sessions cycling long-history turns through a pool too small for them: the cache's hit rate under eviction,
    and the counters of what the pool did, with the pages the cache names sampled through the run. Each session
    has a system block of its own, so no pages are shared between them. One session is run alone first to measure
    the pages it takes, and enough run together to hold twice the pool."""
    s = phase.server
    if s.metric(QUERIED) is None:
        phase.check("the prefix-cache counters are served (--enable-metrics)", False)
        return
    pool = counters(s)["kv_total"]
    base = counters(s)["kv_used"]
    with Sampler(s) as alone:
        conversation(s, LONG_SCRIPTS["plain"], f"session 0: {SYSTEM}", False)
        time.sleep(2.5)
    per_session = max(1.0, max(p["kv_used"] for p in alone.series) - base)
    sessions = max(4, min(64, math.ceil(2 * pool / per_session)))
    before = settled(s)
    with Sampler(s, 3.0) as sampler, ThreadPoolExecutor(sessions) as pool_threads:
        futures = [pool_threads.submit(conversation, s, LONG_SCRIPTS["plain" if i % 2 else "tool_round"],
                                       f"session {i + 1}: {SYSTEM}", False) for i in range(sessions)]
        results = [f.result() for f in futures]
    after = settled(s)
    series = sampler.series
    turns = [row for rows in results for row in rows]
    phase.check("every turn of every session was served", all(r["served"] for r in turns),
                [r["error"] for r in turns if not r["served"]][:3])
    asked, held = after[QUERIED] - before[QUERIED], after[HELD] - before[HELD]
    used_max = max((p["kv_used"] for p in series), default=0.0)
    summary = {"pool_pages": pool, "pages_one_session": per_session, "sessions": sessions, "turns": len(turns),
               "prompt_tokens": asked, "cached_tokens": held, "hit_rate": round(held / asked, 4) if asked else None,
               "preemptions": after[PREEMPTIONS] - before[PREEMPTIONS], "drops": after[DROPS] - before[DROPS],
               "rebuild_tokens": after[REBUILD_TOKENS] - before[REBUILD_TOKENS],
               "rebuild_cached_pages": after[REBUILD_CACHED] - before[REBUILD_CACHED],
               "pages_cached_max": max((p["kv_cached"] for p in series), default=None), "pages_used_max": used_max,
               "snapshot_counters": "none exist: orphaned snapshots are not counted"}
    (phase.out / "eviction_series.json").write_text(json.dumps({"summary": summary, "series": series}, indent=2))
    print(f"  eviction: hit rate {summary['hit_rate']}, preemptions {summary['preemptions']}, "
          f"drops {summary['drops']}, LLM pages used up to {used_max:.0f} of {pool:.0f}")
    phase.check("the pool was small enough to be pressed (preempted, or full to 90%)",
                summary["preemptions"] > 0 or used_max >= 0.9 * pool, summary)


REBUILD_TOOLS = [
    f"Use python to print {i} * {i + 3}, then write a long, detailed explanation of how that product is computed, "
    f"in at least 600 words." for i in range(1, 13)]
REBUILD_SAY = ["good morning to you all", "the weather is fine today", "please read me the first note",
               "what time does the archive open"]
REBUILD_TOKENS_MAX = 1000
REBUILD_JOBS = [("tool", i) for i in range(len(REBUILD_TOOLS))] + [("spoken", i) for i in range(len(REBUILD_SAY))]


def rebuild_request(s: Server, kind: str, i: int, clips: Path, tag: str) -> dict:
    kwargs = {**GREEDY, "max_output_tokens": REBUILD_TOKENS_MAX, "return_token_ids": True}
    # the client's request id comes back in the server's log lines, which say which requests the pool dropped
    rid = f"rb-{tag}-{kind}{i}"
    if kind == "tool":
        r = s.generate(REBUILD_TOOLS[i], kwargs=kwargs, timeout=600, request_id=rid)
    else:
        r = s.generate(wav=clips / f"clip_{i}.wav", kwargs=kwargs, timeout=600, request_id=rid)
    return {"kind": kind, "i": i, "request_id": rid, "served": served(r), "error": r["error"],
            "token_ids": r.get("token_ids", []), "text": r["text"], "heard": r["heard"]}


def run_rebuild_jobs(s: Server, clips: Path, tag: str) -> list[dict]:
    """All the requests at once, so the batches have a shape like the constrained server's."""
    with ThreadPoolExecutor(len(REBUILD_JOBS)) as pool:
        return list(pool.map(lambda job: rebuild_request(s, *job, clips, tag), REBUILD_JOBS))


def first_difference(a: list[int], b: list[int]) -> int | None:
    return next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                None if len(a) == len(b) else min(len(a), len(b)))


def phase_rebuild_reference(phase: Phase) -> None:
    """The answers the `rebuild` phase is compared with: the same requests, all at once, on the unconstrained server,
    twice (`rebuild_answers.json`, `rebuild_answers_again.json`), so the run-to-run variance of 1000 greedy bf16
    tokens under batching is on record. One request of its own alone first, to measure the pages a long answer
    takes; `rebuild_sizing.json` gives the `preempt` arm a pool that the 16 at once overrun."""
    s, clips = phase.server, phase.out / "clips"
    clips.mkdir(exist_ok=True)
    for i, said in enumerate(REBUILD_SAY):
        spoke = s.generate(f"Say '{said}' out loud using the speak tool.", speak=True,
                           kwargs={**GREEDY, "max_output_tokens": 200})
        phase.check(f"clip {i} made", served(spoke) and spoke["audio"] > 0, spoke["error"])
        pcm_wav(clips / f"clip_{i}.wav", spoke["pcm"])
    base = counters(s)["kv_used"]
    # a request of its own kind (a system tag no job shares), so the jobs' prefixes are not warmed by it
    with Sampler(s) as alone:
        s.generate(
            "Use python to print 17 * 19, then write a long, detailed explanation of how that product is computed, "
            "in at least 600 words.",
            history=[{"role": "system", "content": f"sizing run {random.Random().randrange(10**9)}"}],
            kwargs={**GREEDY, "max_output_tokens": REBUILD_TOKENS_MAX}, timeout=600)
        time.sleep(2.5)
    pages = max(1.0, max(p["kv_used"] for p in alone.series) - base)
    sizing = {"pages_per_request": pages,
              "preempt_pages": max(PREEMPT_FLOOR, math.ceil(0.35 * len(REBUILD_JOBS) * pages))}
    (phase.out / "rebuild_sizing.json").write_text(json.dumps(sizing, indent=2))
    phase.check("the pool for the preempt arm is sized from one long request's pages", True, sizing)
    for name, tag in (("rebuild_answers.json", "ref1"), ("rebuild_answers_again.json", "ref2")):
        answers = run_rebuild_jobs(s, clips, tag)
        phase.check(f"every request of {name} was served", all(r["served"] for r in answers),
                    [r["error"] for r in answers if not r["served"]][:3])
        (phase.out / name).write_text(json.dumps(answers, indent=2))


def well_formed(r: dict) -> str | None:
    """Why an answer is not a whole one, or None: served, not empty, and (a tool request) the tool call was made and
    an answer follows it."""
    if not r["served"]:
        return f"not served: {r['error']}"
    if not r["token_ids"] or not r["text"].strip():
        return "empty"
    if r["kind"] == "tool":
        answer = r["text"].rpartition("</tool_call>")
        if not answer[1]:
            return "no tool call"
        if not answer[2].strip():
            return "nothing after the tool call"
    return None


def phase_rebuild(phase: Phase) -> None:
    """The reference's requests, all at once on a pool the long answers outgrow, so the pool drops and rebuilds some.

    Required: drops happened; every request is served and its answer whole (a tool call made and an answer after it).
    Not required: the same tokens as the reference. Greedy bf16 over 1000 tokens differs between two runs of a
    server under batching, which the reference run twice shows, so token-exactness is judged only for a request that
    matched itself (the short spoken ones). That a rebuild writes what the request had written is pinned on the CPU
    (`test_worker_rebuild`, `test_rebuild_policy`); here the rebuild runs on the GPU, in a real batch.
    The first differing token against the reference and against the reference's rerun, per request, are in
    `rebuild_compare.json`, with the requests the server's log says were dropped."""
    s, clips = phase.server, phase.out / "clips"
    paths = [phase.out / "rebuild_answers.json", phase.out / "rebuild_answers_again.json"]
    if not all(p.exists() for p in paths):
        phase.check("the reference is there (run rebuild_reference first, in the same --out)", False)
        return
    ref, again = ({(r["kind"], r["i"]): r for r in json.loads(p.read_text())} for p in paths)
    before = settled(s)
    got = run_rebuild_jobs(s, clips, "pool")
    after = settled(s)
    drops = after[DROPS] - before[DROPS]
    phase.check("the pool forced drops (so some answers below are rebuilt requests)", drops > 0,
                {"drops": drops, "preemptions": after[PREEMPTIONS] - before[PREEMPTIONS],
                 "rebuild_tokens": after[REBUILD_TOKENS] - before[REBUILD_TOKENS], "pool": after["kv_total"]})
    dropped = s.dropped_requests()
    rows = []
    for r in got:
        key = (r["kind"], r["i"])
        vs_ref = first_difference(r["token_ids"], ref[key]["token_ids"])
        vs_rerun = first_difference(again[key]["token_ids"], ref[key]["token_ids"])
        rows.append({"request": r["request_id"], "rebuilt": r["request_id"] in dropped,
                     "first_difference_vs_reference": vs_ref, "reference_vs_its_rerun": vs_rerun,
                     "tokens": len(r["token_ids"])})
        why = well_formed(r)
        phase.check(f"{r['kind']} request {r['i']} is whole" + (" (rebuilt)" if rows[-1]["rebuilt"] else ""),
                    why is None, why)
        if r["kind"] == "spoken" and vs_rerun is None:
            # it matched itself, so the pool run must match too
            phase.check(f"spoken request {r['i']} answers token for token as the reference does",
                        vs_ref is None and r["heard"] == ref[key]["heard"], rows[-1])
    (phase.out / "rebuild_compare.json").write_text(json.dumps(rows, indent=2))
    phase.check("per request: first difference from the reference, and the reference's own (information)", True, rows)


PHASES = {"parity": (phase_parity, ("default", "pre_plan")), "paths": (phase_paths, ("default",)),
          "load": (phase_load, ("default",)), "multiturn": (phase_multiturn, ("default",)),
          "faults": (phase_faults, ("small_kv",)),
          "flashstate": (phase_flashstate, ("default", "messages")),
          "diverge": (phase_diverge, ("default", "messages")), "subagents": (phase_subagents, ("default",)),
          "eviction": (phase_eviction, ("pressure",)),
          "rebuild_reference": (phase_rebuild_reference, ("default",)), "rebuild": (phase_rebuild, ("preempt",))}


# -- launching an arm -----------------------------------------------------------------------------------------------


def arm_config(arm: str, dest: Path) -> Path:
    if yaml is None:
        raise SystemExit("--launch needs pyyaml")
    config = yaml.safe_load(CONFIG.read_text())
    arm_changes = dict(ARMS[arm])
    sizing = dest / "rebuild_sizing.json"
    if arm == "preempt" and sizing.exists():
        arm_changes["resources.kv_cache.max_num_pages"] = json.loads(sizing.read_text())["preempt_pages"]
    for dotted, value in arm_changes.items():
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
    server.group, server.out, server.log = process.pid, dest, dest / f"server_{arm}.log"
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"the {arm} server exited with {process.returncode}; see server_{arm}.log")
        if server.healthy():
            return process, server
        time.sleep(5)
    stop(process)
    raise SystemExit(f"the {arm} server did not come up")


def stop(process: subprocess.Popen) -> None:
    """End the server and its workers, by the process group the harness started (not by name)."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("stress_results"))
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--launch", action="store_true", help="start a server per arm")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--phases", default=",".join(PHASES))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = [p for p in args.phases.split(",") if p]
    results = []
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
    (args.out / "summary.json").write_text(json.dumps(
        [{"phase": r["phase"], "ok": r["ok"], "seconds": r["seconds"]} for r in results], indent=2))
    failed = [r["phase"] for r in results if not r["ok"]]
    print("FAILED: " + ", ".join(failed) if failed else "all phases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
