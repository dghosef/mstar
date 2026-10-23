import torch, flashinfer
from triton.testing import do_bench_cudagraph
for bs in (1, 8, 32):
    q = torch.randn(bs, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(bs, 2, 128, device="cuda", dtype=torch.bfloat16)
    pos = torch.randint(0, 2000, (bs,), device="cuda", dtype=torch.int32)
    cache = torch.randn(65536, 128, device="cuda", dtype=torch.float32)
    t1 = do_bench_cudagraph(lambda: flashinfer.rope.apply_rope_pos_ids_inplace(q, k, pos, rope_theta=1e6), rep=50) * 1e3
    t2 = do_bench_cudagraph(lambda: flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(pos, q.view(bs, -1), k.view(bs, -1), 128, cache, is_neox=True), rep=50) * 1e3
    print(f"bs={bs}: pos_ids {t1:.2f}us  cos_sin_cache {t2:.2f}us")
