"""GPU time per step of _transcribe for LibriSpeech-like steps of 16, at several FORWARD_COST_SAMPLES."""
import sys, time, random, json, torch, soundfile as sf
sys.path.insert(0, "/root/work/mstar")
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size="300m"); sub = m.get_submodule("encoder", "cuda")
durs = [sf.info("/root/work/ls_clean/" + json.loads(l)["audio_path"]).duration for l in open("/root/work/ls_clean/manifest.jsonl")]
random.seed(1)
steps = [[torch.randn(int(16000 * d), device="cuda", dtype=sub.dtype) for d in random.sample(durs, 16)] for _ in range(12)]
with torch.no_grad():
    for cost_s in [10**6, 0, 5, 15, 30, 60]:
        sub.FORWARD_COST_SAMPLES = int(cost_s * 16000) if cost_s < 10**6 else 0
        if cost_s == 10**6:  # greedy (original): one group while within budget
            orig = type(sub).sub_batches
            def greedy(self, lengths):
                groups = []
                for row in sorted(range(len(lengths)), key=lengths.__getitem__):
                    if groups and (len(groups[-1]) + 1) * lengths[row] <= self._max_batch_samples: groups[-1].append(row)
                    else: groups.append([row])
                return groups
            sub.sub_batches = greedy.__get__(sub)
        else:
            sub.sub_batches = type(sub).sub_batches.__get__(sub)
        for st in steps[:3]: sub._transcribe(st)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for st in steps: sub._transcribe(st)
        torch.cuda.synchronize()
        ng = sum(len(sub.sub_batches([w.shape[0] for w in st])) for st in steps) / len(steps)
        print(f"cost {cost_s}s: {1e3*(time.perf_counter()-t0)/len(steps):.1f} ms/step, groups/step {ng:.1f}", flush=True)
