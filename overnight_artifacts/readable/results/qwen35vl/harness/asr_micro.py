import sys, time, torch
sys.path.insert(0, "/root/work/base")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
from mstar.model.omnilingual_asr.components.wav2vec2 import attention_mask
m = OmnilingualASRModel(size="300m")
sub = m.get_submodule("encoder", "cuda")
model = sub.model
lens = [int(16000 * s) for s in (3.1, 4.5, 5.2, 6.0, 6.8, 7.4, 8.0, 9.1, 10.3, 11.0, 12.5, 13.3, 14.0, 15.2, 16.8, 19.0)]
L = max(lens)
batch = torch.randn(16, L, device="cuda", dtype=torch.bfloat16)
fe = model.encoder_frontend.feature_extractor
def bench(fn, *a, n=10):
    for _ in range(3): fn(*a)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn(*a)
    torch.cuda.synchronize(); return 1e3 * (time.perf_counter() - t0) / n
with torch.no_grad():
    print("features eager", bench(fe, batch))
    fec = torch.compile(fe, dynamic=False)
    print("features compiled static", bench(fec, batch))
    x = fe(batch)
    print("feat shape", tuple(x.shape))
    fr = model.frame_lengths(lens); t = x.shape[1]
    mask = torch.arange(t, device="cuda")[None] < torch.tensor(fr, device="cuda")[:, None]
    front = model.encoder_frontend
    def enc(x, mask):
        h = front.model_dim_proj(front.post_extract_layer_norm(x))
        h = front.pos_encoder(h, mask)
        return model.encoder(h, attention_mask(mask))
    def enc_nomask(x):
        h = front.model_dim_proj(front.post_extract_layer_norm(x))
        h = front.pos_encoder(h, None)
        return model.encoder(h, None)
    print("encoder eager masked", bench(enc, x, mask))
    print("encoder eager nomask", bench(enc_nomask, x))
    fed = torch.compile(fe, dynamic=True)
    t0 = time.perf_counter(); fed(batch); torch.cuda.synchronize(); print("dyn compile first call s", time.perf_counter() - t0)
    print("features compiled dynamic", bench(fed, batch))
    import random
    random.seed(0)
    t0 = time.perf_counter()
    for i in range(20):
        b = torch.randn(random.randint(1, 16), random.randint(16000, 16000 * 20), device="cuda", dtype=torch.bfloat16)
        fed(b)
    torch.cuda.synchronize(); print("20 random shapes dyn total s", time.perf_counter() - t0)
    print("features compiled dynamic after", bench(fed, batch))
    def enc_c(x, mask):
        return enc(x, mask).argmax(-1)
    encc = torch.compile(enc_c, dynamic=True)
    t0 = time.perf_counter(); encc(x, mask); torch.cuda.synchronize(); print("enc dyn compile first s", time.perf_counter() - t0)
    print("encoder compiled dyn masked", bench(encc, x, mask))
