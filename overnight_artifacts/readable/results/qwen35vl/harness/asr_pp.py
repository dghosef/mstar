import sys, json, time, torch
sys.path.insert(0, sys.argv[1])
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel
m = OmnilingualASRModel(size="300m")
items = [json.loads(l) for l in open("/root/work/ls_clean/manifest.jsonl")][:100]
paths = ["/root/work/ls_clean/" + it["audio_path"] for it in items]
tl = tp = tpost = 0
ids = torch.randint(0, 50, (400,))
for p in paths:
    t0 = time.perf_counter(); a = m.load_audio(p, "cpu").data; t1 = time.perf_counter()
    o = m.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [a]}); t2 = time.perf_counter()
    m.postprocess(ids, "text"); t3 = time.perf_counter()
    tl += t1 - t0; tp += t2 - t1; tpost += t3 - t2
n = len(paths)
print(f"load {1e3*tl/n:.2f} ms  process_prompt {1e3*tp/n:.2f} ms  postprocess {1e3*tpost/n:.2f} ms")
