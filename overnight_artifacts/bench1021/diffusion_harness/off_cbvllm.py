#!/usr/bin/env python3
"""Offline competitor for Chatterbox (base): chatterbox-vllm @8630314 (randombk; vLLM batches the T3 decode,
then S3Gen runs per prompt; no server). Settings matched to M*'s request: built-in voice (audio_prompt_path
None = the checkpoint's conds.pt, M*'s voice "default"), temperature 0.8, exaggeration 0.5, CFG 0.5 (env
CHATTERBOX_CFG_SCALE), top_p 1.0, repetition_penalty 1.2, 10 diffusion steps; no watermark (M* is sent
watermark false). It has no min_p (M* 0.05) and runs eager by default (--compile = its vLLM CUDA graphs).
Batch b = prompts per generate() call; TTFA = batch latency (no streaming). Run from the repo dir (it links
./t3-model there) with VLLM_ENABLE_V1_MULTIPROCESSING=0.
"""
import io
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doff_common as D  # noqa: E402


def extra(ap):
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.4)


def main():
    a = D.parse(extra)
    import numpy as np
    import torch
    from chatterbox_vllm.tts import ChatterboxTTS
    bmax = max(int(x) for x in a.batch_sizes.split(","))
    kw = {"compile": True} if a.compile else {}
    model = ChatterboxTTS.from_pretrained(max_batch_size=bmax, max_model_len=1200,
                                          gpu_memory_utilization=a.gpu_memory_utilization, **kw)
    sr = model.sr

    def wav(x):
        x = np.clip(np.asarray(x, dtype=np.float32).reshape(-1), -1, 1)
        b = io.BytesIO()
        with wave.open(b, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((x * 32767).astype("<i2").tobytes())
        return b.getvalue(), len(x) / sr

    def process(batch):
        wavs = model.generate([it["text"] for _, it, _ in batch], audio_prompt_path=None, exaggeration=0.5,
                              temperature=0.8, top_p=1.0, repetition_penalty=1.2, diffusion_steps=10)
        return [wav(w.cpu().numpy() if hasattr(w, "cpu") else w) for w in wavs]

    D.run(a, process, sync=torch.cuda.synchronize, ext="wav")


if __name__ == "__main__":
    main()
