"""Host enqueue time vs GPU time of _transcribe over random server-like batches."""
import sys, time, random, torch
sys.path.insert(0, sys.argv[1])
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
from mstar.model.omnilingual_asr import audio
m = OmnilingualASRModel(size="300m")
sub = m.get_submodule("encoder", "cuda")
random.seed(0)
batches = []
for i in range(40):
    n = random.randint(8, 16)
    batches.append([torch.randn(int(16000 * random.uniform(2, 20)), device="cuda", dtype=sub.dtype) for _ in range(n)])
with torch.no_grad():
    for b in batches[:5]:
        sub._transcribe(b)
    torch.cuda.synchronize()
    host, tot = [], []
    s0 = torch.cuda.memory_stats()
    for b in batches[5:]:
        t0 = time.perf_counter(); sub._transcribe(b); t1 = time.perf_counter(); torch.cuda.synchronize(); t2 = time.perf_counter()
        host.append(t1 - t0); tot.append(t2 - t0)
    s1 = torch.cuda.memory_stats()
print(f"host enqueue mean {1e3*sum(host)/len(host):.1f} ms, total {1e3*sum(tot)/len(tot):.1f} ms; "
      f"alloc_retries {s1['num_alloc_retries']-s0['num_alloc_retries']} device_frees {s1.get('num_device_free',0)-s0.get('num_device_free',0)} device_allocs {s1.get('num_device_alloc',0)-s0.get('num_device_alloc',0)}")
from torch.profiler import profile, ProfilerActivity
with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
    for b in batches[5:10]:
        sub._transcribe(b)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18, max_name_column_width=60))
