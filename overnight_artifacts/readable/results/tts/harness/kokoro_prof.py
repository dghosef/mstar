"""Kokoro GPU profile: where a batched step's GPU time goes.

    CUDA_VISIBLE_DEVICES=1 python kokoro_prof.py [--dtype float32|bfloat16] [--compile]
Builds the synth submodule, makes chunks from real sentences via process_prompt, and
times + profiles encode_text and decode_frames (per frame bucket group) eagerly.
"""
import argparse
import collections
import sys

import torch

sys.path.insert(0, "/root/work/exp8")

from mstar.model.kokoro.kokoro_model import KokoroModel  # noqa: E402
from mstar.model.kokoro.submodules import group_by_bucket, pick_bucket  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="float32")
ap.add_argument("--no-compile", action="store_true")
ap.add_argument("--bs", type=int, default=22)
ap.add_argument("--benchmark", action="store_true")
ap.add_argument("--no-profile", action="store_true")
ap.add_argument("--no-fuse", action="store_true")
ap.add_argument("--nlc", action="store_true", help="store decoder conv weights channels-last")
a = ap.parse_args()
torch.backends.cudnn.benchmark = a.benchmark

model = KokoroModel("hexgrad/Kokoro-82M", decoder_dtype=a.dtype, compile_decoder=not a.no_compile,
                    fuse_lstm_directions=not a.no_fuse)
sub = model.get_submodule("kokoro", device="cuda")
if a.nlc:
    with torch.no_grad():
        for mod in sub.model.decoder.modules():
            if isinstance(mod, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                w = mod.weight
                mod.weight.data = w.unsqueeze(2).contiguous(memory_format=torch.channels_last).squeeze(2)
cfg = sub.config
lines = [ln.strip() for ln in open("/root/work/tharness/s26.txt") if ln.strip()]
ids, lens, styles = [], [], []
for ln in lines:
    out = model.process_prompt(ln, ["text"], ["audio"])
    ph, pl, st = out["phoneme_ids"][0], out["phoneme_lens"][0], out["ref_style"][0]
    for i in range(ph.shape[0]):
        ids.append(ph[i, : pl[i]])
        lens.append(int(pl[i]))
        styles.append(st[i])
order = list(range(len(ids)))[: a.bs]
T = pick_bucket(max(lens[i] for i in order), cfg.text_buckets)
input_ids = torch.full((len(order), T), 0, dtype=torch.long)
for r, i in enumerate(order):
    input_ids[r, : lens[i]] = ids[i]
input_ids = input_ids.cuda()
lengths = torch.tensor([lens[i] for i in order]).cuda()
style = torch.stack([styles[i] for i in order]).cuda().float()
speed = torch.ones(len(order)).cuda()
m = sub.model


def timed(fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


with torch.inference_mode():
    d, t_en, pred_dur = m.encode_text(input_ids, lengths, style, speed)
    sizes = pred_dur.sum(1).tolist()
    print("rows", len(order), "T", T, "frames", sizes)
    torch.save({"d": d, "t_en": t_en, "pred_dur": pred_dur}, f"/root/work/kokoro_enc_{'nofuse' if a.no_fuse else 'fuse'}.pt")
    print("encode_text ms", round(timed(lambda: m.encode_text(input_ids, lengths, style, speed)), 2))
    groups = group_by_bucket(sizes, cfg.frame_buckets)
    total = 0.0
    calls = []
    for bucket, rows in groups:
        idx = torch.tensor(rows).cuda()
        en, asr, fl = m.align(d[idx], t_en[idx], pred_dur[idx], bucket)
        st = style[idx]
        ms = timed(lambda: m.decode_frames(en, asr, fl, st))
        total += ms
        calls.append((bucket, len(rows), round(ms, 2)))
        print("decode bucket", bucket, "rows", len(rows), "ms", round(ms, 2))
    print("decode total ms", round(total, 2), "audio s", sum(sizes) * cfg.samples_per_frame / 24000)
    if a.no_profile:
        sys.exit(0)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        m.encode_text(input_ids, lengths, style, speed)
        torch.cuda.synchronize()
    agg, cnt = collections.Counter(), collections.Counter()
    for ev in p.events():
        if ev.device_type.name == "CUDA":
            agg[ev.name[:90]] += ev.device_time_total
            cnt[ev.name[:90]] += 1
    tot = sum(agg.values())
    print(f"profile encode_text total kernel us {tot:.0f} kernels {sum(cnt.values())}")
    for k, v in agg.most_common(12):
        print(f"{100 * v / tot:5.1f}% {v:9.0f}us x{cnt[k]:4d} {k}")
    # profile the biggest group
    bucket, rows = max(groups, key=lambda g: (g[0] or 0) * len(g[1]))
    idx = torch.tensor(rows).cuda()
    en, asr, fl = m.align(d[idx], t_en[idx], pred_dur[idx], bucket)
    st = style[idx]
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        m.decode_frames(en, asr, fl, st)
        torch.cuda.synchronize()
    agg = collections.Counter()
    cnt = collections.Counter()
    for ev in p.events():
        if ev.device_type.name == "CUDA":
            agg[ev.name[:90]] += ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
            cnt[ev.name[:90]] += 1
    tot = sum(agg.values())
    print(f"profile bucket {bucket} rows {len(rows)} total kernel us {tot:.0f} kernels {sum(cnt.values())}")
    for k, v in agg.most_common(25):
        print(f"{100 * v / tot:5.1f}% {v:9.0f}us x{cnt[k]:4d} {k}")
