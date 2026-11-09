#!/usr/bin/env python3
"""Open-loop A/B of worker scheduling policies (the ``batch_builder`` config key).

Replays one arrival trace against a fresh server per arm. Arrivals are decided up front
from the seed and fired on schedule whatever the server is doing, so a slow arm builds a
queue instead of slowing its own load, which a closed-loop test hides. Every arm of a
round sees the same trace. Rounds repeat the arms in rotated order, and the spread
between rounds of one arm is the noise figure to read each difference against.

    python -m benchmark.sched_policy_bench --preset qwen_0.8b --arrivals poisson \
        --rps 40 --duration 60 --arms fifo,hrrn --rounds 3 --out results/q08_poisson

``--url`` skips starting servers and sends one round per arm to a server already up,
which is how the harness itself is checked without a GPU.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import statistics
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
import yaml

WORDS = (
    "river stone market engine lantern orbit quiet copper meadow signal harbor velvet "
    "thunder pencil garden mirror canyon ribbon falcon winter"
).split()


@dataclass(frozen=True)
class RequestClass:
    name: str
    # share of arrivals
    weight: float
    prompt_words: int
    max_tokens: int


@dataclass(frozen=True)
class Preset:
    serve: str
    config: str
    path: str
    served_name: str
    classes: tuple[RequestClass, ...]
    # what a request is judged by for goodput: first output, then per output unit
    ttft_slo_ms: float
    tpot_slo_ms: float


PRESETS = {
    # Short chat next to long-context prompts: where a short request's first token can
    # wait behind a long prefill.
    "qwen_0.8b": Preset(
        "qwen3_5_0.8b", "qwen3_5_0.8b.yaml", "/v1/chat/completions", "Qwen/Qwen3.5-0.8B",
        (RequestClass("short", 0.8, 60, 64), RequestClass("long", 0.2, 3000, 128)),
        ttft_slo_ms=500, tpot_slo_ms=50,
    ),
    # Short and long texts: the first audio chunk plays the part of the first token.
    "qwen3_tts": Preset(
        "qwen3_tts_1p7b", "qwen3tts_1p7b.yaml", "/v1/audio/speech",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        (RequestClass("short", 0.8, 12, 0), RequestClass("long", 0.2, 90, 0)),
        ttft_slo_ms=800, tpot_slo_ms=200,
    ),
    # A TTS model whose prompts do run under a token budget, unlike Qwen3-TTS.
    "orpheus": Preset(
        "orpheus", "orpheus_colocated.yaml", "/v1/audio/speech", "canopylabs/orpheus-3b-0.1-pretrained",
        (RequestClass("short", 0.8, 12, 0), RequestClass("long", 0.2, 90, 0)),
        ttft_slo_ms=800, tpot_slo_ms=200,
    ),
}


@dataclass(frozen=True)
class Arrival:
    at: float
    cls: str
    prompt_words: int
    max_tokens: int
    seed: int


def arrival_times(kind: str, rps: float, duration: float, rng: random.Random) -> list[float]:
    """When each request fires. ``bursty`` keeps the same mean rate as ``poisson``:
    2 s on at 3x the mean, 4 s quiet, so the on periods overload and the quiet ones drain."""
    times: list[float] = []
    t = 0.0
    if kind == "poisson":
        while True:
            t += rng.expovariate(rps)
            if t >= duration:
                return times
            times.append(t)
    if kind != "bursty":
        raise ValueError(f"unknown arrival process {kind!r}")
    on, off, on_rate = 2.0, 4.0, 3 * rps
    while t < duration:
        end = min(t + on, duration)
        at = t
        while True:
            at += rng.expovariate(on_rate)
            if at >= end:
                break
            times.append(at)
        t += on + off
    return times


def build_trace(preset: Preset, kind: str, rps: float, duration: float, seed: int) -> list[Arrival]:
    """The same list for every arm of a round."""
    rng = random.Random(seed)
    classes, weights = preset.classes, [c.weight for c in preset.classes]
    trace = []
    for at in arrival_times(kind, rps, duration, rng):
        cls = rng.choices(classes, weights)[0]
        trace.append(Arrival(at, cls.name, cls.prompt_words, cls.max_tokens, rng.randrange(1 << 31)))
    return trace


def body_for(preset: Preset, arrival: Arrival) -> dict:
    rng = random.Random(arrival.seed)
    # a random start keeps prompts from sharing a prefix, so the prefix cache is out of the picture
    text = " ".join(rng.choice(WORDS) for _ in range(arrival.prompt_words))
    if preset.path == "/v1/audio/speech":
        return {
            "model": preset.served_name, "input": text, "voice": "Vivian", "language": "English",
            "response_format": "pcm", "stream": True,
        }
    return {
        "model": preset.served_name, "messages": [{"role": "user", "content": text}],
        "max_tokens": arrival.max_tokens, "temperature": 0.7, "stream": True, "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }


async def send(session: aiohttp.ClientSession, url: str, preset: Preset, arrival: Arrival, start: float) -> dict:
    """Fire one request at its scheduled time and time its stream."""
    await asyncio.sleep(max(0.0, start + arrival.at - time.perf_counter()))
    sent = time.perf_counter()
    out = {"cls": arrival.cls, "at": arrival.at, "ok": False, "late_ms": (sent - start - arrival.at) * 1e3}
    is_speech = preset.path == "/v1/audio/speech"
    first = last = None
    units = 0
    try:
        async with session.post(url + preset.path, json=body_for(preset, arrival)) as resp:
            if resp.status != 200:
                out["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return out
            if is_speech:
                async for chunk in resp.content.iter_any():
                    last = time.perf_counter()
                    first = first or last
                    units += len(chunk)
            else:
                usage = None
                async for raw in resp.content:
                    line = raw.strip()
                    if not line.startswith(b"data:") or line == b"data: [DONE]":
                        continue
                    event = json.loads(line[5:])
                    usage = event.get("usage") or usage
                    choices = event.get("choices") or ()
                    if choices and (choices[0].get("delta") or {}).get("content"):
                        last = time.perf_counter()
                        first = first or last
                        units += 1
                if usage:
                    units = usage.get("completion_tokens", units)
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        out["error"] = repr(exc)
        return out
    if first is None:
        out["error"] = "no output"
        return out
    out.update(
        ok=True, units=units, ttft_ms=(first - sent) * 1e3, e2e_ms=(last - sent) * 1e3,
        # a one-unit reply has no gap to measure
        tpot_ms=(last - first) * 1e3 / (units - 1) if units > 1 else None,
    )
    return out


async def replay(url: str, preset: Preset, trace: list[Arrival]) -> tuple[list[dict], float]:
    timeout = aiohttp.ClientTimeout(total=None, sock_read=300)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        start = time.perf_counter()
        results = await asyncio.gather(*(send(session, url, preset, a, start) for a in trace))
        return list(results), time.perf_counter() - start


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def latency(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values), "p50": percentile(values, 0.5), "p99": percentile(values, 0.99),
    }


def summarize(results: list[dict], wall: float, preset: Preset) -> dict:
    """Latency over the whole trace and per class, throughput and goodput at the SLO.

    Goodput counts only output from requests that met both SLOs. ``late_ms`` is how far
    behind schedule the client fired: a large p99 there means the load generator itself
    was the bound and the run says nothing about the server."""
    ok = [r for r in results if r["ok"]]

    def block(rows: list[dict]) -> dict:
        tpots = [r["tpot_ms"] for r in rows if r.get("tpot_ms") is not None]
        return {
            "n": len(rows), "ttft_ms": latency([r["ttft_ms"] for r in rows]), "tpot_ms": latency(tpots),
            "e2e_ms": latency([r["e2e_ms"] for r in rows]),
        }

    def met(r: dict) -> bool:
        return r["ttft_ms"] <= preset.ttft_slo_ms and (r.get("tpot_ms") or 0.0) <= preset.tpot_slo_ms

    out = {
        "requests": len(results), "failed": len(results) - len(ok), "wall_s": wall,
        "errors": sorted({r["error"] for r in results if not r["ok"]})[:3],
        "client_late_p99_ms": percentile([r["late_ms"] for r in results], 0.99) if results else None,
        "throughput": sum(r["units"] for r in ok) / wall if wall else 0.0,
        "goodput": sum(r["units"] for r in ok if met(r)) / wall if wall else 0.0,
        "slo_met": sum(map(met, ok)) / len(results) if results else 0.0,
        "all": block(ok),
    }
    for cls in sorted({r["cls"] for r in results}):
        out[cls] = block([r for r in ok if r["cls"] == cls])
    return out


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            return resp.status == 200
    except OSError:
        return False


def arm_config(tree: Path, preset: Preset, arm: str, directory: Path) -> Path:
    """The preset's serving config with only the policy under test changed."""
    config = yaml.safe_load((tree / "configs" / preset.config).read_text())
    config["batch_builder"] = arm
    path = directory / f"{arm}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def start_server(
    serve: str, tree: Path, preset: Preset, config: Path, port: int, log: Path, boot_s: float,
) -> subprocess.Popen:
    with log.open("w") as out:
        proc = subprocess.Popen(
            [*serve.split(), preset.serve, "--config", str(config), "--port", str(port)],
            cwd=tree, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
        )
    deadline = time.monotonic() + boot_s
    while not healthy(port):
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode} during boot, see {log}")
        if time.monotonic() > deadline:
            stop_server(proc)
            raise SystemExit(f"server not healthy after {boot_s:.0f} s, see {log}")
        time.sleep(2)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(30)


def rotated(arms: list[str], round_index: int) -> list[str]:
    """Each round starts one arm later, so no arm always runs first on a cold box."""
    shift = round_index % len(arms)
    return arms[shift:] + arms[:shift]


def table(runs: list[dict], metrics: list[tuple[str, str]]) -> str:
    """One row per metric: each arm's mean over rounds with its min..max across rounds."""
    arms = list(dict.fromkeys(r["arm"] for r in runs))
    lines = ["| metric | " + " | ".join(f"{arm} mean (min..max)" for arm in arms) + " |"]
    lines.append("|---|" + "---|" * len(arms))

    def read(summary: dict, path: str) -> float | None:
        node = summary
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        return node

    for label, path in metrics:
        cells = []
        for arm in arms:
            values = [v for r in runs if r["arm"] == arm if (v := read(r["summary"], path)) is not None]
            if values:
                cells.append(f"{statistics.fmean(values):.1f} ({min(values):.1f}..{max(values):.1f})")
            else:
                cells.append("n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


METRICS = [
    ("TTFT mean ms", "all.ttft_ms.mean"), ("TTFT p50 ms", "all.ttft_ms.p50"), ("TTFT p99 ms", "all.ttft_ms.p99"),
    ("short TTFT mean ms", "short.ttft_ms.mean"), ("short TTFT p99 ms", "short.ttft_ms.p99"),
    ("long TTFT mean ms", "long.ttft_ms.mean"), ("long TTFT p99 ms", "long.ttft_ms.p99"),
    ("TPOT mean ms", "all.tpot_ms.mean"), ("TPOT p99 ms", "all.tpot_ms.p99"),
    ("throughput units/s", "throughput"), ("goodput units/s", "goodput"), ("SLO met", "slo_met"),
    ("failed", "failed"), ("client late p99 ms", "client_late_p99_ms"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--arrivals", default="poisson", choices=("poisson", "bursty"))
    ap.add_argument("--rps", type=float, required=True, help="mean arrival rate")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of arrivals")
    ap.add_argument("--warmup", type=float, default=20.0, help="seconds of the same load, discarded, per server")
    ap.add_argument("--arms", default="fifo", help="comma-separated batch_builder values; repeat fifo for an A/A")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=18500)
    ap.add_argument("--tree", type=Path, default=Path.cwd(), help="checkout to serve from")
    ap.add_argument("--serve", default="mstar serve", help="command that serves a model, before its name")
    ap.add_argument("--boot-s", type=float, default=900.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default=None, help="an already running server: no boot, one round per arm")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    args.out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    runs: list[dict] = []
    for round_index in range(1 if args.url else args.rounds):
        # one trace per round, shared by its arms: arms compare within a round, rounds add spread
        trace = build_trace(preset, args.arrivals, args.rps, args.duration, args.seed + round_index)
        warmup = build_trace(preset, args.arrivals, args.rps, args.warmup, args.seed + 1000 + round_index)
        for position, arm in enumerate(rotated(arms, round_index)):
            name = f"r{round_index}_{position}_{arm}"
            proc = None
            url = args.url
            with tempfile.TemporaryDirectory() as tmp:
                if url is None:
                    config = arm_config(args.tree, preset, arm, Path(tmp))
                    log = args.out / f"{name}.server.log"
                    proc = start_server(args.serve, args.tree, preset, config, args.port, log, args.boot_s)
                    url = f"http://127.0.0.1:{args.port}"
                try:
                    if warmup:
                        asyncio.run(replay(url, preset, warmup))
                    results, wall = asyncio.run(replay(url, preset, trace))
                finally:
                    if proc is not None:
                        stop_server(proc)
            summary = summarize(results, wall, preset)
            run = {"arm": arm, "round": round_index, "position": position, "summary": summary}
            runs.append(run)
            (args.out / f"{name}.json").write_text(json.dumps({**run, "requests": results}))
            print(f"{name}: failed {summary['failed']} TTFT {summary['all']['ttft_ms']}", flush=True)
    report = table(runs, METRICS)
    (args.out / "report.md").write_text(report + "\n")
    (args.out / "meta.json").write_text(json.dumps({**vars(args), "preset_def": asdict(preset)}, default=str, indent=1))
    print(report)


if __name__ == "__main__":
    main()
