"""The `reference` phase of the voice agent's stress harness: each member's served output against its own reference.

``stress.py`` collects the served outputs (``reference_served.json``, the clips and the speech as wavs in
``reference_wavs/``) while the server is up, stops the server, and then ``compare`` loads the references on the freed
GPU, one model at a time, and writes ``reference_compare.json``. Every load is from the HF cache (HF_HUB_OFFLINE=1,
local_files_only). It needs the ``kokoro`` and ``scipy`` packages beside mstar's own install (``pip install kokoro
scipy``); `preflight` checks them before the run starts. The LLM, ASR and TTS judgements are separate checks: one that
fails to run does not hide the others.

Pass criteria, fixed before any run (the mstar-correctness skill's: a tolerance for bf16 near-ties, not exact-only):

LLM, Qwen/Qwen3.5-9B. 8 typed prompts, greedy, at most 128 tokens: 4 plain, 2 that call the python tool (compared up
to and including the call's closing tag), 2 multi-turn (the history the agent built, a tool result appended; the final
answer compared). The reference is HF transformers, the same model in bf16, the same chat template (the deployment's
rendering of earlier turns included), the same tools, ``enable_thinking=False``. The served tokens are teacher-forced
through it, so every position is judged against the oracle's own distribution given the served prefix, and the first
disagreement is where plain greedy decoding would first have diverged. Required:
  - every completion is non-empty, and a tool-call prompt's holds a closed call;
  - mean agreement (the served token is the oracle's argmax) >= 0.95 over all positions;
  - agreement 1.0 at every position where the oracle's top-2 logit margin is > 1.0 (a bf16 near-tie cannot flip those);
  - at each request's first disagreement the served token is within 1.0 logit of the oracle's top and ranks below 5.
ASR, openai/whisper-large-v3-turbo. The 4 spoken clips (the agent's own speech for CLIP_SAY, sent back in). The
agent's transcript tokens are teacher-forced through HF transformers Whisper (bf16, on the same audio, the forced prompt
with the language the reference detects, which must be English) and judged by the same four rules as the LLM's; and
the agent's transcript must equal the reference's greedy one word for word (case and punctuation ignored), clip by
clip. The word error rate over the four is reported, not gated.
TTS, hexgrad/Kokoro-82M. 3 sentences said through the speak tool, against the ``kokoro`` package (KPipeline, the voice
and speed the agent used). Per sentence, required:
  - the duration within 5% of the reference's;
  - the mean log-mel distance along a DTW path (`mel_distance`) <= 0.5 nats. This tolerance was not calibrated
    against a measured spread before the first run: the distance to another sentence's speech is reported beside it
    as the scale of a wrong output, and if the number proves wrong the report says so and keeps this verdict next to
    the new one;
  - Whisper (the ASR reference) hears the same words in both.
The wavs of both are in ``reference_wavs/`` to listen to.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
import wave
from pathlib import Path

import numpy as np

LLM_MODEL = "Qwen/Qwen3.5-9B"
ASR_MODEL = "openai/whisper-large-v3-turbo"
TTS_REPO = "hexgrad/Kokoro-82M"
TTS_DEFAULT_VOICE = "af_heart"   # the voice the agent speaks in (Kokoro's default_voice)
TTS_SPEED = 1.0
TTS_RATE = 24000
LLM_MAX_TOKENS = 128

# the four rules of the LLM and the ASR (the skill's gate.py defaults)
LLM_MEAN_AGREEMENT = 0.95
LLM_CONFIDENT_MARGIN = 1.0
LLM_FIRST_GAP = 1.0
LLM_FIRST_RANK = 5
TTS_DURATION = 0.05
TTS_MEL = 0.5

PLAIN = [
    "What is the capital of France? Answer in one sentence.",
    "Give me three tips for sleeping better, as a short list.",
    "Explain in two sentences what a hash table is.",
    "Translate 'good morning, how are you?' into Spanish.",
]
# a prompt that calls python, and the result its multi-turn history carries back
PYTHON = [
    ("Use python to compute 17 * 23, and tell me the result.", "391"),
    ("Use python to compute the sum of the integers from 1 to 100, and tell me the result.", "5050"),
]
CLIP_SAY = ["good morning to you all", "the weather is fine today", "please read me the first note",
            "what time does the archive open"]
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please remember to turn off the lights before you leave the office tonight.",
    "In nineteen sixty nine, two astronauts walked on the moon, and millions of people watched.",
]
TOOL_CLOSE = "</tool_call>"


def say_prompt(text: str) -> str:
    return f"Say '{text}' out loud using the speak tool."


def spoken_text(reply: str) -> tuple[str | None, str | None]:
    """The text the reply's speak calls carry, joined with a space as the tool node does, and the first call's voice
    (None for either when there is no call or no voice)."""
    calls = re.findall(r"<function=speak>(.*?)</function>", reply, re.DOTALL)

    def parameter(call: str, name: str) -> str | None:
        found = re.search(rf"<parameter={name}>\n?(.*?)\n?</parameter>", call, re.DOTALL)
        return found.group(1) if found else None
    texts = [t.strip() for t in (parameter(c, "text") for c in calls) if t and t.strip()]
    return (" ".join(texts) or None), (parameter(calls[0], "voice") if calls else None)


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        new = [i]
        for j, h in enumerate(hypothesis, 1):
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (r != h)))
        row = new
    return row[-1]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return data, w.getframerate()


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def mel_distance(a: np.ndarray, b: np.ndarray, rate: int = TTS_RATE) -> float:
    """The mean absolute difference of two clips' log-mel frames (natural log of the power, 80 mels) along the
    dynamic-time-warping path that minimises it, so a small difference in pace does not count as a difference in
    sound."""
    import torch
    from transformers.audio_utils import mel_filter_bank

    filters = torch.tensor(mel_filter_bank(
        num_frequency_bins=513, num_mel_filters=80, min_frequency=0.0, max_frequency=rate / 2, sampling_rate=rate,
        norm=None, mel_scale="htk"), dtype=torch.float32)

    def log_mel(x: np.ndarray) -> np.ndarray:
        spec = torch.stft(torch.from_numpy(x), 1024, 256, window=torch.hann_window(1024), return_complex=True)
        return torch.log(filters.T @ spec.abs().pow(2) + 1e-5).T.numpy()

    x, y = log_mel(a), log_mel(b)
    cost = np.abs(x[:, None, :] - y[None, :, :]).mean(-1)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = min(acc[i - 1, j] + cost[i - 1, j - 1], acc[i, j - 1] + cost[i - 1, j - 1],
                            acc[i - 1, j - 1] + 2 * cost[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def free() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()




def preflight() -> None:
    """Stop before a run starts when the references cannot load: the packages beside mstar's install."""
    missing = [name for name in ("kokoro", "scipy", "transformers", "torch") if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"the reference phase needs {missing} installed (pip install kokoro scipy), and the GPU free")


def wait_for_gpu(check, needed: float = 0.9, seconds: float = 90.0) -> None:
    """Wait until the GPU's memory is free of the stopped server, then say how much is."""
    import time

    import torch

    deadline = time.monotonic() + seconds
    while True:
        free_bytes, total = torch.cuda.mem_get_info()
        if free_bytes >= needed * total or time.monotonic() > deadline:
            break
        time.sleep(2)
    check(f"the GPU is free for the references ({free_bytes / 2**30:.1f} of {total / 2**30:.1f} GiB)",
          free_bytes >= needed * total)


# -- the first four rules, for the LLM and the ASR ------------------------------------------------------------------


def score(logits, ids: list[int]) -> dict:
    """The oracle's ``logits`` at the positions that predict ``ids`` (one row each), judged by the four rules: whether
    the served token is the argmax, the oracle's top-2 margin there, and at the first disagreement how far the served
    token is below the top and what rank it holds."""
    import torch

    logits = logits.float()
    top = logits.topk(2, dim=-1).values
    served = torch.tensor(ids, device=logits.device)
    served_logit = logits.gather(-1, served[:, None])[:, 0]
    rank = (logits > served_logit[:, None]).sum(-1)
    gap, margin = top[:, 0] - served_logit, top[:, 0] - top[:, 1]
    agree = rank == 0
    confident = margin > LLM_CONFIDENT_MARGIN
    first = None if bool(agree.all()) else int((~agree).nonzero()[0])
    return {"positions": len(ids), "agreed": int(agree.sum()), "confident": int(confident.sum()),
            "confident_disagreements": int((confident & ~agree).sum()), "first_disagreement": first,
            "first_gap": None if first is None else float(gap[first]),
            "first_rank": None if first is None else int(rank[first])}


def first_is_near_tie(row: dict) -> bool:
    return row["first_disagreement"] is None or (
        row["first_gap"] <= LLM_FIRST_GAP and row["first_rank"] < LLM_FIRST_RANK)


def judge_totals(label: str, rows: list[dict], check) -> None:
    positions = sum(r["positions"] for r in rows)
    mean = sum(r["agreed"] for r in rows) / max(positions, 1)
    confident = sum(r["confident"] for r in rows)
    bad = sum(r["confident_disagreements"] for r in rows)
    check(f"{label} mean agreement over {positions} positions >= {LLM_MEAN_AGREEMENT}",
          mean >= LLM_MEAN_AGREEMENT, mean)
    check(f"{label} agreement is 1.0 at the {confident} positions where the oracle's margin > {LLM_CONFIDENT_MARGIN}",
          bad == 0, bad)


# -- the LLM ------------------------------------------------------------------------------------------------------


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """The deployment's rendering: the chat template with the three tools, thinking off, and every earlier assistant
    turn written as it was generated (Qwen3_5DenseModel._exact_history)."""
    from mstar.model.components.python_tool import SPAWN, SPEAK, TOOL_SCHEMAS
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    only_after = Qwen3_5DenseModel._THINK_ONLY_AFTER_QUERY
    assert only_after in tokenizer.chat_template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        tools=[TOOL_SCHEMAS[name] for name in ("python", SPEAK, SPAWN)],
        chat_template=tokenizer.chat_template.replace(
            only_after, "{%- if loop.index0 > ns.last_query_index or true %}", 1))
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def cut_after_call(tokenizer, ids: list[int]) -> list[int] | None:
    """``ids`` up to and including the token that closes the first tool call, or None when none closes."""
    for n in range(1, len(ids) + 1):
        if TOOL_CLOSE in tokenizer.decode(ids[:n]):
            return ids[:n]
    return None


def judge_llm(served: list[dict], check, out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, local_files_only=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            LLM_MODEL, dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    except (ValueError, KeyError):
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    model.eval()
    rows = []
    for item in served:
        messages = item.get("history") or [{"role": "user", "content": item["prompt"]}]
        ids = item["token_ids"]
        if item["kind"] == "python":
            cut = cut_after_call(tokenizer, ids)
            check(f"python prompt {item['index']}: the served reply closes a tool call", cut is not None,
                  item["text"][:200])
            ids = cut or ids
        label = f"{item['kind']} {item['index']}"
        check(f"{label}: a non-empty completion", len(ids) > 0, item.get("error"))
        if not ids:
            continue
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([prompt_ids(tokenizer, messages) + ids], device="cuda"),
                           logits_to_keep=len(ids) + 1).logits[0, :-1]
        row = {"kind": item["kind"], "index": item["index"], **score(logits, ids), "served": item["text"][:300]}
        rows.append(row)
        check(f"{label}: the first disagreement is a near-tie (or there is none)", first_is_near_tie(row), row)
    del model
    free()
    judge_totals("LLM", rows, check)
    (out / "reference_llm.json").write_text(json.dumps(rows, indent=2))


# -- ASR and TTS --------------------------------------------------------------------------------------------------


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    g = np.gcd(rate, 16000)
    return resample_poly(audio, 16000 // g, rate // g).astype(np.float32)


def load_whisper():
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(ASR_MODEL, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    return processor, model


def features(processor, audio: np.ndarray, rate: int):
    import torch

    return processor(to_16k(audio, rate), sampling_rate=16000, return_tensors="pt").input_features.to(
        "cuda", torch.bfloat16)


def transcribe(clips: list[tuple[np.ndarray, int]]) -> list[str]:
    """HF Whisper's greedy transcript of each clip, the language detected."""
    import torch

    processor, model = load_whisper()
    texts = []
    for audio, rate in clips:
        with torch.no_grad():
            ids = model.generate(features(processor, audio, rate), task="transcribe", do_sample=False, num_beams=1,
                                 max_new_tokens=440)
        texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    del model
    free()
    return texts


def judge_asr(asr: list[dict], check, out: Path) -> None:
    import torch

    folder = out / "reference_wavs"
    processor, model = load_whisper()
    tokenizer = processor.tokenizer
    forced = [tokenizer.convert_tokens_to_ids(t) for t in
              ("<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>")]
    rows, errors, total = [], 0, 0
    for item in asr:
        audio, rate = read_wav(folder / f"clip_{item['index']}.wav")
        x = features(processor, audio, rate)
        ids = item.get("heard_ids") or []
        label = f"ASR clip {item['index']}"
        check(f"{label}: the agent's transcript has tokens", len(ids) > 0, item["heard"])
        with torch.no_grad():
            language = model.detect_language(input_features=x)
            greedy = model.generate(x, task="transcribe", do_sample=False, num_beams=1, max_new_tokens=440)
            reference = processor.batch_decode(greedy, skip_special_tokens=True)[0].strip()
            logits = model(input_features=x, decoder_input_ids=torch.tensor([forced + ids[:-1]], device="cuda")).logits[
                0, len(forced) - 1:] if ids else None
        detected = int(language.flatten()[0])
        check(f"{label}: the reference detects English", detected == forced[1], detected)
        row = {"index": item["index"], "said": item["said"], "agent_heard": item["heard"], "reference_heard": reference}
        if logits is not None:
            row.update(score(logits, ids))
            check(f"{label}: the first disagreement is a near-tie (or there is none)", first_is_near_tie(row), row)
        r, h = words(reference), words(item["heard"])
        errors, total = errors + word_errors(r, h), total + len(r)
        check(f"{label}: the agent's transcript equals the reference's word for word", r == h, row)
        rows.append(row)
    del model
    free()
    judge_totals("ASR", [r for r in rows if "positions" in r], check)
    (out / "reference_asr.json").write_text(
        json.dumps({"wer_information": errors / max(total, 1), "rows": rows}, indent=2))


def reference_speech(texts: list[tuple[str, str]]) -> list[np.ndarray]:
    """The ``kokoro`` package's speech for each (text, voice), at the agent's speed."""
    import torch
    from kokoro import KModel, KPipeline

    model = KModel(repo_id=TTS_REPO).to("cuda").eval()
    pipelines: dict[str, KPipeline] = {}
    audio = []
    for text, voice in texts:
        if voice[0] not in pipelines:
            pipelines[voice[0]] = KPipeline(lang_code=voice[0], repo_id=TTS_REPO, model=model)
        with torch.no_grad():
            parts = [r.audio.cpu().numpy() for r in pipelines[voice[0]](text, voice=voice, speed=TTS_SPEED)
                     if r.audio is not None]
        audio.append(np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32))
    del model, pipelines
    free()
    return audio


def judge_tts(tts: list[dict], check, out: Path) -> None:
    folder = out / "reference_wavs"
    served_speech = [read_wav(folder / f"tts_{i}_served.wav") for i in range(len(tts))]
    voices = [(t["spoken"] or t["sentence"], t["voice"] or TTS_DEFAULT_VOICE) for t in tts]
    reference = reference_speech(voices)
    for i, audio in enumerate(reference):
        write_wav(folder / f"tts_{i}_reference.wav", audio, TTS_RATE)
    heard = transcribe([*served_speech, *[(a, TTS_RATE) for a in reference]])
    served_words, reference_words = heard[:len(tts)], heard[len(tts):]
    rows = []
    for i, item in enumerate(tts):
        served_audio, rate = served_speech[i]
        ratio = (len(served_audio) / rate) / max(len(reference[i]) / TTS_RATE, 1e-9)
        distance = mel_distance(served_audio, reference[i])
        other = mel_distance(served_audio, reference[(i + 1) % len(tts)]) if len(tts) > 1 else None
        rows.append({"sentence": item["sentence"], "spoken": item["spoken"], "duration_ratio": ratio,
                     "mel_distance": distance, "mel_distance_to_another_sentence": other,
                     "served_heard": served_words[i], "reference_heard": reference_words[i]})
        row = rows[-1]
        check(f"TTS {i}: the agent spoke the sentence", words(item["spoken"] or "") == words(item["sentence"]),
              item["spoken"])
        check(f"TTS {i}: duration within {TTS_DURATION:.0%} of the reference ({ratio:.3f})",
              abs(ratio - 1) <= TTS_DURATION, row)
        check(f"TTS {i}: mel distance <= {TTS_MEL} ({distance:.3f}; another sentence's: {other})",
              distance <= TTS_MEL, row)
        check(f"TTS {i}: Whisper hears the same words in both",
              words(served_words[i]) == words(reference_words[i]), row)
    (out / "reference_tts.json").write_text(json.dumps(rows, indent=2))


def compare(out: Path, check) -> None:
    """Judge ``<out>/reference_served.json`` against the references: the LLM, then the ASR, then the TTS, each its own
    check when it fails to run. Needs the GPU free: stop the server first."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    served = json.loads((out / "reference_served.json").read_text())
    wait_for_gpu(check)
    for name, judge, data in (("LLM", judge_llm, served["llm"]), ("ASR", judge_asr, served["asr"]),
                              ("TTS", judge_tts, served["tts"])):
        try:
            judge(data, check, out)
        except Exception as error:  # noqa: BLE001  a judge that cannot run is a result, and the others still run
            check(f"the {name} comparison ran to the end", False, f"{type(error).__name__}: {error}")
            free()
