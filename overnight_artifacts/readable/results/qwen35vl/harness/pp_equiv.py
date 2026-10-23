"""pixel_values from base tree vs head tree must be bitwise equal; also time both."""
import json, sys, time, importlib, torch
torch.set_num_threads(4)
items = [it for it in json.load(open("/root/work/imgset/manifest.json")) if it["image"]]
def run(tree):
    for k in [k for k in sys.modules if k.startswith("mstar")]:
        del sys.modules[k]
    sys.path.insert(0, tree)
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    sys.path.pop(0)
    m = Qwen3_5DenseModel("Qwen/Qwen3.5-0.8B", cache_dir=None)
    outs, times = {}, {}
    for rep in range(3):
        for it in items:
            t0 = time.perf_counter()
            img = m.load_image(it["image"], "cpu").data
            o = m.process_prompt("What is this?", ["image", "text"], ["text"], tensors={"image_inputs": [img]})
            times.setdefault(it["kind"], []).append(time.perf_counter() - t0)
            outs[it["id"]] = o
    return outs, {k: 1e3 * sum(v) / len(v) for k, v in times.items()}
a, ta = run("/root/work/base")
b, tb = run("/root/work/mstar")
bad = 0
for k in a:
    for key in ("pixel_values", "image_grid_thw", "text_inputs"):
        for x, y in zip(a[k][key], b[k][key]):
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                bad += 1; print("DIFF", k, key)
print("diffs", bad, "base ms", ta, "head ms", tb, "head image_inputs", {k: len(v.get("image_inputs", [None])) for k, v in list(b.items())[:2]})
