"""A command-line client for the voice agent (configs/voice_agent.yaml).

Start the server first:  mstar serve configs/voice_agent.yaml

    python examples/voice_agent/client.py "What is 17 * 23? Use python."          # a typed turn
    python examples/voice_agent/client.py --wav question.wav --speak              # a spoken turn, spoken answer
    python examples/voice_agent/client.py --chat                                  # a conversation, turn by turn

It posts each turn to /generate and prints, as the chunks arrive: what the agent heard (the transcript of a spoken
turn), the reply, the tool calls and agents the reply starts, and where the audio was saved. The conversation so far
is sent with every turn (the ``history`` model kwarg), and extended with the reply.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import wave
from pathlib import Path

import requests

# the signals of the stream (a chunk's ``metadata["signal"]``): what Whisper says, as text
HEARD = "asr_text"
SAMPLE_RATE = 24000   # Kokoro's, headerless 16-bit mono PCM

TOOL_CALL = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def describe_calls(text: str) -> list[str]:
    """The tool calls in a reply's text, as one line each."""
    lines = []
    for name, body in TOOL_CALL.findall(text):
        args = {key: value for key, value in PARAM.findall(body)}
        shown = ", ".join(f"{key}={value[:60]!r}" for key, value in args.items())
        lines.append(f"{name}({shown})")
    return lines


def turn(server: str, text: str | None, wav: Path | None, history: list[dict], speak: bool, thinking: bool,
         max_tokens: int, timeout: float, request_id: str | None = None):
    """Post one turn; yield (modality, signal, bytes) per chunk as they arrive."""
    model_kwargs = {"history": history, "enable_thinking": thinking, "max_output_tokens": max_tokens}
    data = {
        "output_modalities": "text,audio" if speak else "text",
        "model_kwargs": json.dumps(model_kwargs),
    }
    if request_id:
        data["request_id"] = request_id
    files = None
    if text:
        data["text"] = text
    if wav is not None:
        files = [("files", (wav.name, wav.read_bytes(), "audio/wav"))]
    with requests.post(f"{server}/generate", data=data, files=files, stream=True, timeout=timeout) as response:
        if response.status_code != 200:
            raise SystemExit(f"server answered {response.status_code}: {response.text[:500]}")
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                signal = (chunk.get("metadata") or {}).get("signal", "")
                yield chunk["modality"], signal, base64.b64decode(chunk["data"])


def save_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)


def run_turn(args, text: str | None, wav: Path | None, history: list[dict]) -> tuple[str, str]:
    started = time.monotonic()
    heard, reply, pcm, first = [], [], bytearray(), None
    seen_calls = 0
    for modality, signal, payload in turn(
        args.server, text, wav, history, args.speak, args.thinking, args.max_tokens, args.timeout,
    ):
        first = first if first is not None else time.monotonic() - started
        if modality == "audio":
            pcm += payload
        elif modality == "text" and signal == HEARD:
            heard.append(payload.decode("utf-8", "replace"))
            print(f"\rheard: {''.join(heard).strip()}", end="", flush=True)
        elif modality == "text":
            reply.append(payload.decode("utf-8", "replace"))
            print(reply[-1], end="", flush=True)
            calls = describe_calls("".join(reply))
            for call in calls[seen_calls:]:
                print(f"\n  [{'agent' if call.startswith('spawn_agent') else 'tool'}] {call}", flush=True)
            seen_calls = len(calls)
        elif modality == "error":
            print(f"\n[error] {payload.decode('utf-8', 'replace')}", file=sys.stderr)
    if heard:
        print()
    print(f"\n[{time.monotonic() - started:.1f}s, first chunk {first or 0:.2f}s]")
    if pcm:
        path = Path(args.out)
        save_wav(path, bytes(pcm))
        print(f"[audio: {len(pcm) / 2 / SAMPLE_RATE:.1f}s -> {path}]")
        if args.play:
            import subprocess
            subprocess.run(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", str(path)], check=False)
    return "".join(reply), "".join(heard).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="a typed turn")
    parser.add_argument("--wav", type=Path, help="a spoken turn: a wav file")
    parser.add_argument("--chat", action="store_true", help="read turns from the terminal until EOF")
    parser.add_argument("--speak", action="store_true", help="ask for audio out (the reply may call `speak`)")
    parser.add_argument("--play", action="store_true", help="play the saved audio with ffplay")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--history", type=Path, help="a JSON file of earlier messages [{role, content}, ...]")
    parser.add_argument("--out", default="reply.wav", help="where the audio goes")
    parser.add_argument("--thinking", action="store_true", help="let the model think first (slower)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    history = json.loads(args.history.read_text()) if args.history else []
    if args.chat:
        print("one turn per line; a path ending in .wav is a spoken turn; Ctrl-D ends")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            wav = Path(line) if line.endswith(".wav") and Path(line).exists() else None
            text = None if wav else line
            reply, heard = run_turn(args, text, wav, history)
            history += [{"role": "user", "content": text or heard}, {"role": "assistant", "content": reply}]
        return
    if not args.text and not args.wav:
        parser.error("give a typed turn, --wav, or --chat")
    run_turn(args, args.text, args.wav, history)


if __name__ == "__main__":
    main()
