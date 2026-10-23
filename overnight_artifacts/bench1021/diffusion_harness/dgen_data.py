#!/usr/bin/env python3
"""Items files (JSONL) for the dgen entries; run once per workload in setup (M* PR venv: datasets,
soundfile, PIL). Every item carries "units" (the throughput unit: audio seconds for ASR, frames for
video, 1 otherwise) and the fields its request template and gate need.

  librispeech     LibriSpeech test-clean, every utterance (2620), sorted by id: audio_path, ref, units=duration
  librispeech500  the first 500 of those in a fixed stride order (the PR measured 500)
  parti           64 PartiPrompts (the diffusion lane's parti64.txt): prompt
  vbench          16 VBench prompts (vbench16.txt): prompt
Builders named "<module>:<function>" are looked up in this file's BUILDERS (model-specific ones are
added below the generic ones).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WORK = Path("/root/work")
LS = WORK / "data/ls_audio/LibriSpeech/test-clean"


def librispeech(limit=None):
    import soundfile as sf
    rows = []
    for tr in sorted(LS.glob("*/*/*.trans.txt")):
        for line in tr.read_text().splitlines():
            uid, text = line.split(" ", 1)
            p = tr.parent / f"{uid}.flac"
            if p.exists():
                info = sf.info(str(p))
                rows.append({"id": uid, "audio_path": str(p), "ref": text, "units": info.frames / info.samplerate})
    if not rows:
        raise SystemExit(f"no LibriSpeech audio under {LS}")
    if limit:
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    return rows


def prompts(path):
    return [{"prompt": p.strip(), "units": 1} for p in Path(path).read_text().splitlines() if p.strip()]


MEDIA = WORK / "data/gen/media"
COSMOS_PROMPTS = [  # the overnight video harness's prompts (ovn_video_bench.py), VBench-style
    "a person swimming in ocean", "a person giving a presentation to a room full of colleagues",
    "a person washing the dishes", "a person eating a burger", "a person walking in the snowstorm",
    "a person drinking coffee in a cafe", "a person playing guitar", "a bicycle leaning against a tree",
    "a bicycle gliding through a snowy field", "a bicycle slowing down to stop"]
EDIT_PROMPTS = [
    "Turn it into a watercolor painting.", "Make it look like it is snowing.", "Change the background to a beach at sunset.",
    "Render it in the style of a pencil sketch.", "Make the scene look like night time with warm lamp light.",
    "Add a small red flower in the foreground.", "Make it look like a vintage photograph.", "Turn it into a pixel-art image."]


def data_url(path, mime):
    import base64
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def food101(n):
    """The first n validation images of ethz/food101 (the PR's understanding set), saved as JPEG."""
    from datasets import load_dataset
    d = MEDIA / "food101"
    d.mkdir(parents=True, exist_ok=True)
    paths = sorted(d.glob("*.jpg"))
    if len(paths) < n:
        ds = load_dataset("ethz/food101", split="validation", streaming=True)
        for i, row in enumerate(ds):
            if i >= n:
                break
            row["image"].convert("RGB").save(d / f"{i:03d}.jpg", quality=95)
        paths = sorted(d.glob("*.jpg"))
    return paths[:n]


def bagel_und():
    return [{"image_path": str(p), "image_data_url": data_url(p, "image/jpeg"),
             "prompt": "What food is shown? Answer briefly.", "units": 1} for p in food101(32)]


def bagel_edit():
    """16 food101 images resized to 1024x1024 (every engine then edits on the same canvas) + an instruction."""
    from PIL import Image
    out = []
    for i, p in enumerate(food101(16)):
        q = MEDIA / "food101" / f"edit_{i:03d}.png"
        if not q.exists():
            Image.open(p).convert("RGB").resize((1024, 1024), Image.BICUBIC).save(q)
        out.append({"image_path": str(q), "image_data_url": data_url(q, "image/png"),
                    "prompt": EDIT_PROMPTS[i % len(EDIT_PROMPTS)], "units": 1})
    return out


def cosmos(frames):
    return [{"prompt": p, "units": frames} for p in COSMOS_PROMPTS]


def droid():
    import base64
    from huggingface_hub import hf_hub_download
    img = hf_hub_download("nvidia/Cosmos3-Edge", "assets/example_i2v_input.jpg")
    mk = {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10, "action_chunk_size": 32,
          "num_frames": 33, "seed": 0, "num_inference_steps": 4, "guidance_scale": 3.0, "flow_shift": 5.0,
          "size": "832x480", "fps": 15}
    return [{"image_path": img, "image_b64": base64.b64encode(Path(img).read_bytes()).decode(),
             "task": "Pick up the red cup and place it in the sink.", "model_kwargs": json.dumps(mk), "units": 1}]


LINGBOT_PROMPT = ("A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped "
                  "mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and "
                  "sky, creating a tranquil, meditative atmosphere.")


def lingbot():
    import numpy as np
    ex = WORK / "engines/lingbot-world-v2/examples/03"
    mk = {"seed": 42, "num_frames": 93, "camera_poses": np.load(ex / "poses.npy").tolist(),
          "intrinsics": np.load(ex / "intrinsics.npy")[0].tolist(), "height": 480, "width": 832}
    return [{"image_path": str(ex / "image.jpg"), "example_dir": str(ex), "prompt": LINGBOT_PROMPT,
             "model_kwargs": json.dumps(mk), "h": 480, "w": 832, "units": 93}]


def vjepa(ac):
    v = WORK / "mstar/test/qwen3-omni/video.webm"
    if not ac:
        return [{"video_path": str(v), "model_kwargs": "{}", "units": 1}]
    ramp = [-0.5, -1 / 6, 1 / 6, 0.5]
    st = [0.1, 0.1 + 0.8 / 3, 0.1 + 1.6 / 3, 0.9]
    mk = {"rollout_horizon": 4, "actions": [[a] * 7 for a in ramp], "states": [[x] * 7 for x in st]}
    return [{"video_path": str(v), "model_kwargs": json.dumps(mk), "chunks": 4, "units": 1}]


def waypoint(h, w):
    """Biome's pinned default seed (the image M*'s Waypoint oracle uses), resized to the variant geometry, and
    the 64-step action script of test/waypoint/serve_rollout.py (mouse +-12 alternating, button 0 every 4th)."""
    from PIL import Image
    src = MEDIA / "waypoint_default.jpg"
    if not src.exists():
        import urllib.request
        src.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/Overworldai/Biome/14343a6/seeds/default.jpg", src)
    q = MEDIA / f"waypoint_seed_{w}x{h}.png"
    if not q.exists():
        Image.open(src).convert("RGB").resize((w, h), Image.LANCZOS).save(q)
    acts = [{"mouse": [12.0 if i % 2 else -12.0, 0.0], "buttons": [0] if i % 4 == 0 else [], "scroll": 0.0}
            for i in range(64)]
    mk = {"num_steps": 64, "actions": acts, "seed": 112464007}
    return [{"image_path": str(q), "model_kwargs": json.dumps(mk), "h": h, "w": w, "units": 256}]


BUILDERS = {
    "bagel_und": bagel_und,
    "bagel_edit": bagel_edit,
    "cosmos_nano": lambda: cosmos(189),
    "cosmos_edge": lambda: cosmos(121),
    "droid": droid,
    "lingbot": lingbot,
    "vjepa_vitl": lambda: vjepa(False),
    "vjepa_ac": lambda: vjepa(True),
    "waypoint720": lambda: waypoint(720, 1280),
    "waypoint360": lambda: waypoint(360, 640),
    "librispeech": lambda: librispeech(),
    "librispeech500": lambda: librispeech(500),
    "parti": lambda: prompts(WORK / "data/parti64.txt"),
    "vbench": lambda: prompts(WORK / "data/vbench16.txt"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = BUILDERS[a.builder]()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.rename(out)
    print(f"{a.builder}: {len(rows)} items -> {out}")


if __name__ == "__main__":
    main()
