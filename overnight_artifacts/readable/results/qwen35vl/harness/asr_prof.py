"""Offline kernel profile of the omnilingual CTC encoder at a batch like the server's c=64 steps."""
import sys, json, time, torch
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/root/work/base")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
from mstar.model.omnilingual_asr import audio
size = sys.argv[2] if len(sys.argv) > 2 else "300m"
m = OmnilingualASRModel(size=size)
sub = m.get_submodule("encoder", "cuda")
torch.manual_seed(0)
lens = [int(16000 * s) for s in (3.1, 4.5, 5.2, 6.0, 6.8, 7.4, 8.0, 9.1, 10.3, 11.0, 12.5, 13.3, 14.0, 15.2, 16.8, 19.0)]
waves = [audio.preprocess(torch.randn(n)).cuda().to(sub.dtype) for n in lens]
with torch.no_grad():
    for _ in range(3):
        sub._transcribe(waves)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        sub._transcribe(waves)
    torch.cuda.synchronize()
    print(f"bs16 step: {1e3*(time.perf_counter()-t0)/10:.1f} ms wall")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        sub._transcribe(waves); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
