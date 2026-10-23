#!/usr/bin/env python3
"""Orpheus TTS served by vLLM's in-process AsyncLLM, decoded with SNAC.

A thin, self-contained competitor arm for canopylabs/orpheus-3b-0.1-ft:

* the LLM runs in vLLM (v1 engine, ``vllm.v1.engine.async_llm.AsyncLLM``,
  checked against vLLM v0.30.0 and v0.32.0), bf16, CUDA graphs on, prefix
  caching off;
* generated ids stream back as DELTA token ids (no detokenization) and are
  turned into SNAC codes;
* SNAC (hubertsiuzdak/snac_24khz, fp32) decodes each 28-token / 4-frame
  sliding window and emits samples [2048:4096] of it, exactly as
  orpheus-speech's ``tokens_decoder`` does; windows from all concurrent
  requests are batched into one SNAC forward per window size.

Prompt: [128259] + tok("{voice}: {text}") (BOS included) + [128009, 128260,
128261, 128257], the same ids as orpheus-speech and M*. Generation stops at
128258 (end of speech), as in M* and vox-serve.

Endpoints:
  GET  /health                    200 once the engine is up and healthy
  GET  /v1/audio/voices
  POST /v1/audio/speech           {"input": str, "voice": "tara", "seed": int,
                                   "stream": bool, "response_format": "pcm"|"wav",
                                   optional "temperature", "top_p",
                                   "repetition_penalty", "max_tokens"}
    stream=true : 200 is sent with the first audio chunk; body is raw s16le
                  24 kHz mono PCM ("pcm"), or a 44-byte streaming WAV header
                  followed by PCM ("wav", the default).
    stream=false: one WAV (or raw PCM) blob.

Run (one GPU):
  CUDA_VISIBLE_DEVICES=0 python orpheus_vllm_server.py --port 8200
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import struct
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing, asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("orpheus_vllm_server")

# Orpheus special ids (Llama 3.2 vocab + <custom_token_N> = 128256 + N).
START_OF_HUMAN = 128259
PROMPT_END_IDS = [128009, 128260, 128261, 128257]  # eot, end_of_human, start_of_ai, start_of_speech
END_OF_SPEECH = 128258
AUDIO_TOKEN_BASE = 128256 + 10  # <custom_token_10>: code 0 of frame position 0

TOKENS_PER_FRAME = 7
WINDOW_FRAMES = 4  # 28-token sliding window
HOP = 2048  # samples per frame
SAMPLE_RATE = 24000
VOICES = ["tara", "zoe", "zac", "jess", "leo", "mia", "julia", "leah"]


def wav_header(num_bytes: int | None) -> bytes:
    """44-byte PCM16 mono WAV header; ``None`` writes the open-ended streaming sizes."""
    data = 0xFFFFFFFF if num_bytes is None else num_bytes
    riff = 0xFFFFFFFF if num_bytes is None else 36 + num_bytes
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", riff, b"WAVE", b"fmt ", 16, 1, 1,
        SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b"data", data,
    )


class SnacBatcher:
    """Decodes SNAC windows from every request, batched per window size.

    One decode thread: the event loop keeps reading vLLM outputs while a batch
    runs, and the windows that arrive meanwhile form the next batch.
    """

    def __init__(self, repo: str, device: str, max_batch: int):
        import torch
        from snac import SNAC

        self.torch = torch
        self.device = device
        self.max_batch = max_batch
        self.model = SNAC.from_pretrained(repo).eval().to(device)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snac")
        self.idx_c1 = torch.tensor([1, 4], device=device)
        self.idx_c2 = torch.tensor([2, 3, 5, 6], device=device)

    def submit(self, frames: list[int], start: int, end: int) -> asyncio.Future:
        """Queue ``frames`` (flat codes, a multiple of 7); the future yields samples [start:end] as s16le bytes."""
        fut = asyncio.get_running_loop().create_future()
        self.queue.put_nowait((frames, start, end, fut))
        return fut

    def _decode(self, items: list[tuple[list[int], int, int]]) -> list[bytes]:
        torch = self.torch
        out: list[bytes] = [b""] * len(items)
        groups: dict[int, list[int]] = defaultdict(list)
        for i, (frames, _, _) in enumerate(items):
            groups[len(frames) // TOKENS_PER_FRAME].append(i)
        with torch.inference_mode():
            for width, rows in groups.items():
                codes = torch.tensor(
                    [items[i][0] for i in rows], dtype=torch.int32, device=self.device,
                ).view(len(rows), width, TOKENS_PER_FRAME)
                # orpheus-speech's frame layout: c0 = [f0], c1 = [f1, f4], c2 = [f2, f3, f5, f6]
                c0 = codes[:, :, 0]
                c1 = codes.index_select(2, self.idx_c1).reshape(len(rows), -1)
                c2 = codes.index_select(2, self.idx_c2).reshape(len(rows), -1)
                audio = self.model.decode([c0, c1, c2])  # [B, 1, width * HOP]
                pcm = (audio.squeeze(1).clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
                for j, i in enumerate(rows):
                    _, start, end, *_ = items[i]
                    out[i] = pcm[j, start:end].tobytes()
        return out

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            while len(batch) < self.max_batch and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            try:
                results = await loop.run_in_executor(
                    self.executor, self._decode, [(f, s, e) for f, s, e, _ in batch],
                )
            except Exception as exc:  # surface to every waiting request
                for *_, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (*_, fut), pcm in zip(batch, results):
                if not fut.done():
                    fut.set_result(pcm)

    def warmup(self) -> None:
        zeros = [0] * (WINDOW_FRAMES * TOKENS_PER_FRAME)
        for bs in sorted({1, self.max_batch}):
            self._decode([(zeros, 2 * HOP, 4 * HOP)] * bs)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: str
    voice: str | None = "tara"
    seed: int | None = None
    stream: bool = False
    response_format: str = "wav"
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None


def build_app(args: argparse.Namespace) -> FastAPI:
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The engine first: its EngineCore process starts before this process touches CUDA.
        engine_args = AsyncEngineArgs(
            model=args.model,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=False,
            enforce_eager=args.enforce_eager,
            max_num_seqs=args.max_num_seqs,
            disable_log_stats=not args.log_stats,
        )
        state["engine"] = AsyncLLM.from_engine_args(engine_args)
        state["tokenizer"] = AutoTokenizer.from_pretrained(args.tokenizer)
        batcher = SnacBatcher(args.snac_model, args.snac_device, args.snac_max_batch)
        batcher.warmup()
        state["batcher"] = batcher
        task = asyncio.create_task(batcher.run())
        logger.info("ready: %s on vLLM AsyncLLM, SNAC on %s", args.model, args.snac_device)
        try:
            yield
        finally:
            task.cancel()
            state["engine"].shutdown()

    app = FastAPI(lifespan=lifespan)

    def prompt_ids(text: str, voice: str | None) -> list[int]:
        prompt = f"{voice}: {text}" if voice else text
        return [START_OF_HUMAN, *state["tokenizer"](prompt).input_ids, *PROMPT_END_IDS]

    async def synthesize(req: SpeechRequest, ids: list[int]) -> AsyncIterator[bytes]:
        """Yield s16le PCM, one frame (2048 samples) per new 7-token frame once 4 frames exist."""
        engine, batcher = state["engine"], state["batcher"]
        max_tokens = min(req.max_tokens or args.max_tokens, args.max_model_len - len(ids))
        params = SamplingParams(
            temperature=args.temperature if req.temperature is None else req.temperature,
            top_p=args.top_p if req.top_p is None else req.top_p,
            repetition_penalty=(
                args.repetition_penalty if req.repetition_penalty is None else req.repetition_penalty
            ),
            max_tokens=max_tokens,
            stop_token_ids=[END_OF_SPEECH],
            seed=None if args.ignore_seed else req.seed,
            output_kind=RequestOutputKind.DELTA,
            detokenize=False,
        )
        full_edges = args.edges == "full"
        codes: list[int] = []
        request_id = f"tts-{uuid.uuid4().hex}"
        async with aclosing(engine.generate(TokensPrompt(prompt_token_ids=ids), params, request_id)) as stream:
            async for out in stream:
                windows = []
                for token in out.outputs[0].token_ids:
                    if token < AUDIO_TOKEN_BASE:
                        continue  # end of speech or another marker: not an audio code
                    # code = N - 10 - (position % 7) * 4096 for <custom_token_N>
                    codes.append((token - AUDIO_TOKEN_BASE) % 4096)
                    n = len(codes)
                    if n % TOKENS_PER_FRAME == 0 and n >= WINDOW_FRAMES * TOKENS_PER_FRAME:
                        first = full_edges and n == WINDOW_FRAMES * TOKENS_PER_FRAME
                        windows.append(batcher.submit(
                            codes[-WINDOW_FRAMES * TOKENS_PER_FRAME:], 0 if first else HOP, 2 * HOP,
                        ))
                for fut in windows:
                    yield await fut
        if full_edges:
            frames = len(codes) // TOKENS_PER_FRAME
            if frames >= WINDOW_FRAMES:  # the last two frames, never a window's middle
                yield await batcher.submit(
                    codes[(frames - WINDOW_FRAMES) * TOKENS_PER_FRAME: frames * TOKENS_PER_FRAME],
                    2 * HOP, WINDOW_FRAMES * HOP,
                )
            elif frames:
                yield await batcher.submit(codes[: frames * TOKENS_PER_FRAME], 0, frames * HOP)

    @app.get("/health")
    async def health():
        if "engine" not in state:
            return JSONResponse({"status": "starting"}, status_code=503)
        try:
            await state["engine"].check_health()
        except Exception as exc:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)
        return {"status": "ok"}

    @app.get("/v1/audio/voices")
    async def voices():
        return {"voices": VOICES, "default_voice": "tara"}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        if not req.input.strip():
            raise HTTPException(400, "input must be non-empty")
        voice = req.voice.lower() if req.voice else None
        if voice and voice not in VOICES:
            raise HTTPException(400, f"unsupported voice {voice!r}; supported: {', '.join(VOICES)}")
        if req.response_format not in ("pcm", "wav"):
            raise HTTPException(400, "response_format must be 'pcm' or 'wav'")
        ids = prompt_ids(req.input, voice)
        if len(ids) >= args.max_model_len:
            raise HTTPException(400, f"prompt is {len(ids)} tokens; max_model_len is {args.max_model_len}")
        raw_pcm = req.response_format == "pcm"
        audio = synthesize(req, ids)

        if not req.stream:
            async with aclosing(audio):
                pcm = b"".join([chunk async for chunk in audio])
            body = pcm if raw_pcm else wav_header(len(pcm)) + pcm
            return Response(body, media_type="audio/pcm" if raw_pcm else "audio/wav")

        # Commit to 200 with the first audio chunk, so an engine error is still an HTTP error.
        try:
            first = await anext(audio, None)
        except Exception as exc:
            await audio.aclose()
            raise HTTPException(500, f"generation failed: {exc}") from exc

        async def body_iter() -> AsyncIterator[bytes]:
            async with aclosing(audio):
                if not raw_pcm:
                    yield wav_header(None)
                if first is not None:
                    yield first
                async for chunk in audio:
                    yield chunk

        return StreamingResponse(body_iter(), media_type="audio/pcm" if raw_pcm else "audio/wav")

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8200)
    p.add_argument("--model", default="canopylabs/orpheus-3b-0.1-ft")
    # orpheus-speech's and M*'s prompt tokenizer (same Llama 3.2 vocab as the ft repo)
    p.add_argument("--tokenizer", default="canopylabs/orpheus-3b-0.1-pretrained")
    p.add_argument("--snac-model", default="hubertsiuzdak/snac_24khz")
    p.add_argument("--snac-device", default="cuda:0")
    p.add_argument("--snac-max-batch", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs (on by default)")
    p.add_argument("--log-stats", action="store_true")
    # Sampling defaults: M*'s OrpheusModelConfig (= orpheus-speech's generate_tokens_sync
    # and vox-serve), max tokens = M*'s decode-loop cap (MAX_OUTPUT_TOKENS).
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--ignore-seed", action="store_true",
                   help="drop per-request seeds (vLLM samples seeded rows one by one)")
    p.add_argument("--edges", choices=["reference", "full"], default="reference",
                   help="reference: orpheus-speech's slicing, which never emits the first frame or "
                        "the last two; full: also emit them, so the audio covers every frame like M*")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
