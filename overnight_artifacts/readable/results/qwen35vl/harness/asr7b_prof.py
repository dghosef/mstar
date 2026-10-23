import sys, time, torch
sys.path.insert(0, "/root/work/head4")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
size = sys.argv[1] if len(sys.argv) > 1 else "7b"
m = OmnilingualASRModel(size=size); sub = m.get_submodule("encoder", "cuda")
w = [torch.randn(16000 * 7, device="cuda", dtype=sub.dtype)]
with torch.no_grad():
    for _ in range(3): sub._transcribe(w)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5): sub._transcribe(w)
    torch.cuda.synchronize(); print(f"{size} 7 s clip padded path: {1e3*(time.perf_counter()-t0)/5:.1f} ms")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        sub._transcribe(w); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=80))
