#!/usr/bin/env python3
"""Fused vs unfused CodePredictor depth loop: outputs and captured time.

    PYTHONPATH=<tree> CUDA_VISIBLE_DEVICES=1 python cp_fused_check.py [model]
"""
import sys

import torch

import mstar.model.qwen3_tts.components.talker as T
from mstar.model.registry import get_model_class, model_init_kwargs

name = sys.argv[1] if len(sys.argv) > 1 else "qwen3_tts"
model = get_model_class(name)(**model_init_kwargs(name))
sub = model.get_submodule("Talker", device="cuda:0", autocast_dtype=torch.bfloat16)
cp = sub.code_predictor
cfg = sub.cp_config
dev = "cuda:0"
fused_fn = T.can_fuse_qk_norm_rope_kv


def run(bs, fused, x0):
    T.can_fuse_qk_norm_rope_kv = fused_fn if fused else (lambda qkv: False)
    cache = torch.zeros((cfg.num_hidden_layers, bs, 2, sub.num_codes, cfg.num_key_value_heads, cfg.head_dim),
                        dtype=torch.bfloat16, device=dev)
    pos = torch.zeros(bs, 1, dtype=torch.int32, device=dev)
    outs = []
    x = x0.clone()
    for p in range(sub.num_codes):
        pos.fill_(p)
        h = cp.forward_depth_unrolled(x.clone(), pos, cache, cache_pos=p)
        outs.append(h)
        x = x0 * (0.5 + 0.01 * p)
    return torch.cat(outs, 1), cache


def timed(bs, fused, x0):
    T.can_fuse_qk_norm_rope_kv = fused_fn if fused else (lambda qkv: False)
    for _ in range(2):
        run(bs, fused, x0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(bs, fused, x0)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 20


torch.manual_seed(0)
with torch.inference_mode():
    for bs in (1, 32, 64):
        x0 = torch.randn(bs, 1, sub.talker_config.hidden_size, dtype=torch.bfloat16, device=dev)
        a, ca = run(bs, False, x0)
        b, cb = run(bs, True, x0)
        rel = ((a.float() - b.float()).norm() / a.float().norm()).item()
        kv = ((ca.float() - cb.float()).abs().max()).item()
        print(f"bs={bs} hidden rel_err={rel:.2e} max|dkv|={kv:.3e} "
              f"unfused={timed(bs, False, x0):.3f} ms fused={timed(bs, True, x0):.3f} ms", flush=True)
