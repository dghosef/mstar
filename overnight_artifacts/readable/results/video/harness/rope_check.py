import torch
from mstar.model.wan22.components.dit import Wan22RoPE3D, _apply_rotary_emb


def old(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


rope = Wan22RoPE3D(128, 1024)
cos, sin = rope((21, 22, 40), torch.device("cuda"))
q = torch.randn(2, 21 * 22 * 40, 24, 128, device="cuda", dtype=torch.bfloat16)
print("eager exact:", torch.equal(old(q, cos, sin), _apply_rotary_emb(q, cos, sin)))
co, cn = torch.compile(old), torch.compile(_apply_rotary_emb)
print("compiled exact:", torch.equal(co(q, cos, sin), cn(q, cos, sin)))
for name, f in (("old", co), ("new", cn)):
    for _ in range(3):
        f(q, cos, sin)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(20):
        f(q, cos, sin)
    e.record()
    torch.cuda.synchronize()
    print(name, round(s.elapsed_time(e) / 20, 3), "ms")
