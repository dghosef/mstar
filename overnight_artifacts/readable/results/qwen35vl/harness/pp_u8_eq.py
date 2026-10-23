"""New (byte patches + device normalize) vs base process_prompt pixel_values, bitwise; incl. resized sizes."""
import json, sys, torch, tempfile
from PIL import Image
torch.set_num_threads(4)
def load(tree):
    for k in [k for k in sys.modules if k.startswith("mstar")]:
        del sys.modules[k]
    sys.path.insert(0, tree)
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    from mstar.model.qwen3_5.submodules import VisionEncoderSubmodule
    sys.path.pop(0)
    return Qwen3_5DenseModel("Qwen/Qwen3.5-0.8B"), VisionEncoderSubmodule
items = [it["image"] for it in json.load(open("/root/work/imgset/manifest.json")) if it["image"]]
tmp = tempfile.mkdtemp()
src = Image.open(items[0])
for w, h in ((500, 377), (1001, 333), (3000, 2000), (64, 900)):
    p = f"{tmp}/r_{w}x{h}.png"; src.resize((w, h)).save(p); items.append(p)
base, _ = load("/root/work/base2")
head, VE = load(sys.argv[1])
mean, std = head.pixel_normalizer()
bad = 0
for p in items:
    a = base.process_prompt("x", ["image", "text"], ["text"], tensors={"image_inputs": [base.load_image(p, "cpu").data]})
    b = head.process_prompt("x", ["image", "text"], ["text"], tensors={"image_inputs": [head.load_image(p, "cpu").data]})
    b = b.new_input_tensors if hasattr(b, "new_input_tensors") else b
    pa = a["pixel_values"][0]
    pb = b["pixel_values"][0]
    ve = VE.__new__(VE); ve._pixel_normalizer = (mean, std)
    nb = VE._normalize(ve, pb.cuda()).cpu()
    ok = pb.dtype == torch.uint8 and torch.equal(pa, nb) and torch.equal(a["image_grid_thw"][0], b["image_grid_thw"][0])
    bad += not ok
    if not ok: print("DIFF", p, pa.shape, nb.shape, (pa - nb).abs().max())
print("mismatches", bad, "of", len(items))
