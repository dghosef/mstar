import sys, time, torch
sys.path.insert(0, sys.argv[1])
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size="300m")
t0 = time.perf_counter(); sub = m.get_submodule("encoder", "cuda"); torch.cuda.synchronize()
print(f"load+warm ok {time.perf_counter()-t0:.1f}s")
