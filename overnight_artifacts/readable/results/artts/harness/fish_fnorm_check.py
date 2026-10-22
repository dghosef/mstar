#!/usr/bin/env python3
"""Fast AR step: deferred-residual FlashInfer norms vs the plain block chain (eager)."""
import torch

from mstar.model.registry import get_model_class, model_init_kwargs

model = get_model_class("fish_s2_pro")(**model_init_kwargs("fish_s2_pro"))
sub = model.get_submodule("AR", device="cuda:0")
fa = sub.fast_ar


import copy
f32 = copy.deepcopy(fa).float()


def old_step(x, kv, pos, m=fa):
    f = m.freqs_cis[pos:pos + 1].unsqueeze(0)
    for i, layer in enumerate(m.layers):
        x = layer(x, f, kv[i], pos)
    return m.norm(x)


torch.manual_seed(0)
with torch.inference_mode():
    for bs in (1, 32, 64):
        dt = fa.embeddings.weight.dtype
        kv_a = torch.zeros(fa.kv_cache_shape(bs), dtype=dt, device="cuda:0")
        kv_b = kv_a.clone(); kv_c = kv_a.float()
        h = torch.randn(bs, 2560, dtype=dt, device="cuda:0")
        sem = torch.randint(0, 4096, (bs,), device="cuda:0")
        agree = tot = 0
        rels = []; ea = []; eb = []
        xa = fa.project_in(h).unsqueeze(1)
        a = old_step(xa.clone(), kv_a, 0); b = fa.step(xa.clone(), kv_b, 0); old_step(xa.float(), kv_c, 0, f32)
        x = fa.embeddings(sem).unsqueeze(1)
        for pos in range(1, 10):
            a = old_step(x.clone(), kv_a, pos); b = fa.step(x.clone(), kv_b, pos); c = old_step(x.float(), kv_c, pos, f32)
            ea.append(((a.float() - c).norm() / c.norm()).item()); eb.append(((b.float() - c).norm() / c.norm()).item())
            rels.append(((a.float() - b.float()).norm() / a.float().norm()).item())
            ca = fa.output(a).argmax(-1); cb = fa.output(b).argmax(-1)
            agree += (ca == cb).sum().item(); tot += ca.numel()
            x = fa.embeddings(ca.view(-1)).unsqueeze(1)
        print(f"bs={bs} max rel err {max(rels):.2e}  argmax agree {agree}/{tot}  vs fp32: old {max(ea):.2e} new {max(eb):.2e}", flush=True)
