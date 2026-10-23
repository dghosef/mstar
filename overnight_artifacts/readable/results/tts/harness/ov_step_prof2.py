"""One OmniVoice unmask step at c=1: GPU kernel breakdown of encode + head + scoring.

    CUDA_VISIBLE_DEVICES=1 python ov_step_prof.py
"""
import collections
import sys

import torch

sys.path.insert(0, "/root/work/head4")
from omnivoice.models.omnivoice import OmniVoice  # noqa: E402
from omnivoice.models.omnivoice_flashinfer import apply_flashinfer  # noqa: E402

from mstar.model.omnivoice.components.backbone import CanvasItem, OmniVoiceBackbone, build_packed_canvas  # noqa: E402
from mstar.model.omnivoice.components.unmask import apply_reveal, predict_tokens_with_scoring  # noqa: E402

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", dtype=torch.float16).eval().cuda()
model.llm.to(torch.float16)
apply_flashinfer(model, enable_cuda_graph=False)
bb = OmniVoiceBackbone(model, torch.float16).eval()
g = torch.Generator().manual_seed(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
P, T = (int(x) for x in (sys.argv[2:4] if len(sys.argv) > 3 else (60, 150)))
items = [CanvasItem(str(i), torch.randint(0, 1000, (8, P), generator=g).cuda(), torch.zeros(P, dtype=torch.bool).cuda(),
                    torch.full((1, 8, T), 1024).cuda(), 2.0) for i in range(N)]
item = items[0]
canvas = build_packed_canvas(items, 1024, torch.device("cuda"))
pos = canvas.position_ids[0]


def step():
    hidden = bb.encode(canvas.packed_ids[0].t(), canvas.audio_mask[0], pos, pos.to(torch.int32), attn=model._fi_runner.wrapper)
    logits = bb.head(hidden, canvas.tgt_index)
    for it in items:
        c, u = canvas.slice_logits(logits, it)
        pred, scores = predict_tokens_with_scoring(c.float(), u.float(), 1024, 2.0, 0.0, None)
        apply_reveal(it.tokens.clone(), pred, scores, 40, 1024, 5.0, 5.0, None)


with torch.inference_mode():
    model._fi_runner.plan(canvas.doc_lens, torch.float16)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    import time
    t = time.perf_counter()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    print(f"N={N} P={P} T={T} tokens={canvas.packed_ids.shape} step {(time.perf_counter() - t) * 100:.2f} ms")
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        step()
        torch.cuda.synchronize()
agg, cnt = collections.Counter(), collections.Counter()
for ev in p.events():
    if ev.device_type.name == "CUDA":
        agg[ev.name[:80]] += ev.device_time_total
        cnt[ev.name[:80]] += 1
tot = sum(agg.values())
print(f"total kernel us {tot:.0f} kernels {sum(cnt.values())}")
for k, v in agg.most_common(25):
    print(f"{100 * v / tot:5.1f}% {v:8.0f}us x{cnt[k]:4d} {k}")
