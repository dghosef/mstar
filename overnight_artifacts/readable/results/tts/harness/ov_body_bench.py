"""OmniVoice Qwen3 body: the reference's patched llm vs a layer loop without the q/k/v
split copies (and optionally with fused add+RMSNorm). Same weights, same kernels otherwise.

    CUDA_VISIBLE_DEVICES=1 python ov_body_bench.py N P T
"""
import sys
import time

import flashinfer
import torch

sys.path.insert(0, "/root/work/head4")
from omnivoice.models import omnivoice_flashinfer as fi  # noqa: E402
from omnivoice.models.omnivoice import OmniVoice  # noqa: E402
from omnivoice.models.omnivoice_flashinfer import apply_flashinfer  # noqa: E402

from mstar.model.omnivoice.components.backbone import CanvasItem, OmniVoiceBackbone, build_packed_canvas  # noqa: E402

N, P, T = (int(x) for x in sys.argv[1:4])
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", dtype=torch.float16).eval().cuda()
model.llm.to(torch.float16)
apply_flashinfer(model, enable_cuda_graph=False)
bb = OmniVoiceBackbone(model, torch.float16).eval()
g = torch.Generator().manual_seed(0)
items = []
for i in range(N):
    tok = torch.full((1, 8, T), 1024)
    tok[..., ::3] = torch.randint(0, 1024, tok[..., ::3].shape, generator=g)
    items.append(CanvasItem(str(i), torch.randint(0, 1000, (8, P), generator=g).cuda(),
                            torch.zeros(P, dtype=torch.bool).cuda(), tok.cuda(), 2.0))
canvas = build_packed_canvas(items, 1024, torch.device("cuda"))
pos = canvas.position_ids[0]
pos32 = pos.to(torch.int32)
llm = model.llm
wrapper = model._fi_runner.wrapper


def body(embeds, fused_add):
    """embeds [S, hidden] -> final-normed hidden [S, hidden]."""
    h = embeds
    residual = None
    for layer in llm.layers:
        attn, mlp = layer.self_attn, layer.mlp
        ln1, ln2 = layer.input_layernorm, layer.post_attention_layernorm
        if fused_add and residual is not None:
            flashinfer.norm.fused_add_rmsnorm(h, residual, ln1.weight, ln1.variance_epsilon)
            x = h
        else:
            if residual is not None:
                h = residual + h
            residual = h
            x = flashinfer.norm.rmsnorm(h, ln1.weight, ln1.variance_epsilon)
        s = x.shape[0]
        d = attn.head_dim
        qkv = torch.nn.functional.linear(x, attn._fi_w_qkv).view(s, -1, d)
        nq, nk, _ = (n // d for n in attn._fi_qkv_split)
        q = flashinfer.norm.rmsnorm(qkv[:, :nq], attn.q_norm.weight, attn.q_norm.variance_epsilon)
        k = flashinfer.norm.rmsnorm(qkv[:, nq:nq + nk], attn.k_norm.weight, attn.k_norm.variance_epsilon)
        v = qkv[:, nq + nk:]
        flashinfer.rope.apply_rope_pos_ids_inplace(q, k, pos32, rope_theta=attn._fi_rope_theta, interleave=False)
        o = wrapper.run(q, k, v)
        h = torch.nn.functional.linear(o.reshape(s, -1), attn.o_proj.weight)
        if fused_add:
            flashinfer.norm.fused_add_rmsnorm(h, residual, ln2.weight, ln2.variance_epsilon)
            x = h
        else:
            h = residual + h
            residual = h
            x = flashinfer.norm.rmsnorm(h, ln2.weight, ln2.variance_epsilon)
        y = flashinfer.activation.silu_and_mul(torch.nn.functional.linear(x, mlp._fi_w_gate_up))
        h = torch.nn.functional.linear(y, mlp.down_proj.weight)
    if fused_add:
        flashinfer.norm.fused_add_rmsnorm(h, residual, llm.norm.weight, llm.norm.variance_epsilon)
        return h
    return flashinfer.norm.rmsnorm(residual + h, llm.norm.weight, llm.norm.variance_epsilon)


def ref():
    return bb.encode(canvas.packed_ids[0].t(), canvas.audio_mask[0], pos, pos32, attn=wrapper)


def mine(fused_add):
    embeds = model._prepare_embed_inputs(canvas.packed_ids, canvas.audio_mask)[0]
    return body(embeds, fused_add)


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


with torch.inference_mode():
    model._fi_runner.plan(canvas.doc_lens, torch.float16)
    fi._CTX["doc_slots"] = None
    want = ref()
    a = mine(False)
    b = mine(True)
    print(f"N={N} P={P} T={T} S={canvas.packed_ids.shape[-1]}")
    print("nocopy == ref:", torch.equal(a, want), f"max|d| {(a - want).abs().max().item():.4g}")
    print("fused  vs ref:", f"max|d| {(b - want).abs().max().item():.4g}",
          f"mean|d| {(b - want).abs().float().mean().item():.4g}, ref mean|x| {want.abs().float().mean().item():.4g}")
    hw = bb.head(want, canvas.tgt_index).argmax(-1)
    hb = bb.head(b, canvas.tgt_index).argmax(-1)
    print("fused head argmax agreement:", (hw == hb).float().mean().item())
    for label, fn in (("ref", ref), ("nocopy", lambda: mine(False)), ("fused", lambda: mine(True))):
        print(label, f"{timeit(fn):.2f} ms")
