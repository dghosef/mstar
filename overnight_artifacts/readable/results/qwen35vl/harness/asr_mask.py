import sys, time, torch
sys.path.insert(0, "/root/work/base")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
from mstar.model.omnilingual_asr.components.wav2vec2 import attention_mask, _ATTENTION_BACKENDS
from torch.nn.attention import sdpa_kernel
m = OmnilingualASRModel(size="300m"); model = m.get_submodule("encoder", "cuda").model
lens = [int(16000 * s) for s in (3.1, 4.5, 5.2, 6.0, 6.8, 7.4, 8.0, 9.1, 10.3, 11.0, 12.5, 13.3, 14.0, 15.2, 16.8, 19.0)]
batch = torch.randn(16, max(lens), device="cuda", dtype=torch.bfloat16)
front = model.encoder_frontend
def bench(fn, *a, n=10):
    for _ in range(3): fn(*a)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn(*a)
    torch.cuda.synchronize(); return 1e3 * (time.perf_counter() - t0) / n
with torch.no_grad():
    x = front.feature_extractor(batch); t = x.shape[1]
    fr = model.frame_lengths(lens)
    mask = torch.arange(t, device="cuda")[None] < torch.tensor(fr, device="cuda")[:, None]
    h = front.pos_encoder(front.model_dim_proj(front.post_extract_layer_norm(x)), mask)
    bm = attention_mask(mask)
    fm = torch.zeros(bm.shape, dtype=h.dtype, device="cuda").masked_fill_(~bm, float("-inf"))
    def run(mm):
        with sdpa_kernel(_ATTENTION_BACKENDS):
            return model.encoder(h, mm)
    a = run(bm); b = run(fm)
    print("bool", bench(run, bm), "float", bench(run, fm), "equal", torch.equal(a, b), (a.float()-b.float()).abs().max().item())
    fm2 = fm.expand(-1, 16, -1, -1)
