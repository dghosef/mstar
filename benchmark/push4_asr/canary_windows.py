"""Canary long-form check: cut the long-form WAV into mstar's fixed 40 s windows (the CanaryAdapter's
max_audio_seconds, no balancing or silence search), write them to --out-dir as w<NN>.wav, then transcribe each window
with NeMo (--nemo) or with an mstar server (--url), one line per window: index, words, the window's most repeated
3-gram count (a repetition loop shows up there) and the text's tail.

  nemo/bin/python canary_windows.py --wav long.wav --out-dir win --nemo
  python canary_windows.py --wav long.wav --out-dir win --url http://127.0.0.1:29200
"""
import argparse
import collections
import json
import urllib.request
import uuid
import wave
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--wav", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--seconds", type=float, default=40.0)
ap.add_argument("--route-cuts", default=None, metavar="TREE",
                help="cut as the transcription route does (media_io.split_windows from TREE, 2 s / 20 ms quiet search)")
ap.add_argument("--nemo", action="store_true")
ap.add_argument("--url", default=None)
ap.add_argument("--parallel", action="store_true", help="send every window at once (as the long-form route does)")
ap.add_argument("--whole", action="store_true", help="also send the whole file once (the server windows it)")
a = ap.parse_args()
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
with wave.open(a.wav, "rb") as r:
    params, frames = r.getparams(), r.readframes(r.getnframes())
if a.route_cuts:
    import sys

    import numpy as np

    sys.path.insert(0, a.route_cuts)
    from mstar.api_server.media_io import split_windows

    samples = np.frombuffer(frames, dtype=np.int16)
    pieces = [p.tobytes() for p in split_windows(samples, a.seconds, params.framerate, search_seconds=2.0,
                                                  frame_seconds=0.02)]
else:
    step = int(a.seconds * params.framerate) * params.sampwidth
    pieces = [frames[i:i + step] for i in range(0, len(frames), step)]
paths, t = [], 0.0
for i, piece in enumerate(pieces):
    p = out / f"w{i:02d}.wav"
    with wave.open(str(p), "wb") as w:
        w.setparams(params)
        w.writeframes(piece)
    dur = len(piece) / params.sampwidth / params.framerate
    print(f"w{i:02d} {t:7.2f}-{t + dur:7.2f} s", flush=True)
    t += dur
    paths.append(p)


def loops(text):
    words = text.split()
    grams = collections.Counter(tuple(words[k:k + 3]) for k in range(len(words) - 2))
    return max(grams.values()) if grams else 0


if a.nemo:
    import torch
    from nemo.collections.asr.models import ASRModel

    m = ASRModel.from_pretrained("nvidia/canary-1b-v2").cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hyps = m.transcribe([str(p) for p in paths], batch_size=8, verbose=False, source_lang="en", target_lang="en")
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    texts = [h.text if hasattr(h, "text") else str(h) for h in hyps]
else:
    def post(p):
        boundary = uuid.uuid4().hex
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
            for k, v in (("model", "canary"), ("language", "en"), ("temperature", "0"), ("response_format", "json"))
        ]) + (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{p.name}\"\r\n"
              "Content-Type: audio/wav\r\n\r\n").encode() + p.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{a.url}/v1/audio/transcriptions", body,
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return json.load(urllib.request.urlopen(req, timeout=600))["text"]

    if a.parallel:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(len(paths)) as ex:
            texts = list(ex.map(post, paths))
    else:
        texts = [post(p) for p in paths]
    if a.whole:
        whole = post(Path(a.wav))
        print(f"whole file: words {len(whole.split())} max3gram {loops(whole.upper())}", flush=True)
for i, t in enumerate(texts):
    print(f"w{i:02d} words {len(t.split()):4d} max3gram {loops(t.upper()):3d} | ...{t[-90:]}", flush=True)
