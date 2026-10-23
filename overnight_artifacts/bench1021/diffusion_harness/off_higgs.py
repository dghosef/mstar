#!/usr/bin/env python3
"""Offline competitor for Higgs Audio v3 STT: the official HF remote code (trust_remote_code,
transformers 4.51.0, eager attention as the model card uses, bf16) driven through the repo's own
transcribe.py::transcribe_batch (its batched path), the model's default prompt, enable_thinking=False
(M*'s default; the reference defaults to True), greedy, max_new_tokens 1024, 4 s chunks from the
checkpoint config. Requests are length-sorted within a run; audio is decoded per batch inside the timer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doff_common as D  # noqa: E402


def extra(ap):
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--think", action="store_true")


def main():
    a = D.parse(extra)
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    repo = snapshot_download("bosonai/higgs-audio-v3-stt")
    sys.path.insert(0, repo)
    import transcribe as T  # the repo's own batched transcription
    model = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                      attn_implementation=a.attn, device_map="cuda:0").eval()
    tok = AutoTokenizer.from_pretrained(repo)
    model.audio_out_bos_token_id = tok.convert_tokens_to_ids("<|audio_out_bos|>")
    model.audio_eos_token_id = tok.convert_tokens_to_ids("<|audio_eos|>")
    dur = {}

    def length(item):
        p = item["audio_path"]
        if p not in dur:
            dur[p] = sf.info(p).duration
        return dur[p]

    def order(reqs):
        return sorted(reqs, key=lambda r: -length(a.items_list[r[1]]))

    def load16k(p):
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        return x

    def process(batch):
        auds = [load16k(it["audio_path"]) for _, it, _ in batch]
        return list(T.transcribe_batch(model, tok, auds, sample_rates=16000, enable_thinking=a.think,
                                       max_new_tokens=1024))

    D.run(a, process, sync=torch.cuda.synchronize, order=order)


if __name__ == "__main__":
    main()
