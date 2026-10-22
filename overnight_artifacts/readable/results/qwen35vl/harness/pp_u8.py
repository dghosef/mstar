import json, time, torch, sys
sys.path.insert(0, "/root/work/head2")
torch.set_num_threads(4)
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel, _as_hwc_uint8
m = Qwen3_5DenseModel("Qwen/Qwen3.5-0.8B")
p = m.image_processor
print(type(p).__name__, p.rescale_factor, p.image_mean, p.image_std, p.do_rescale, p.do_normalize)
items = [it for it in json.load(open("/root/work/imgset/manifest.json")) if it["image"]]
bad = 0
for it in items:
    img = m.load_image(it["image"], "cpu").data
    hwc = _as_hwc_uint8(img)
    t0 = time.perf_counter(); ref = p(images=[hwc], return_tensors="pt"); t1 = time.perf_counter()
    u8 = p(images=[hwc], return_tensors="pt", do_rescale=False, do_normalize=False); t2 = time.perf_counter()
    pv = u8["pixel_values"]
    mean = torch.tensor(p.image_mean) * (1.0 / p.rescale_factor)
    std = torch.tensor(p.image_std) * (1.0 / p.rescale_factor)
    # channel of each column: 1536 = C(3) x T(2) x 16 x 16, channel-major
    C = 3; per = pv.shape[1] // C
    mcol = mean.repeat_interleave(per); scol = std.repeat_interleave(per)
    t3 = time.perf_counter(); norm = (pv.to(torch.float32) - mcol) / scol; t4 = time.perf_counter()
    eq = torch.equal(norm, ref["pixel_values"]) and torch.equal(u8["image_grid_thw"], ref["image_grid_thw"])
    bad += not eq
    if it["kind"] in ("huge", "xl", "small") and it["id"].endswith("_0"):
        print(it["id"], pv.dtype, tuple(pv.shape), f"ref {1e3*(t1-t0):.1f}ms u8 {1e3*(t2-t1):.1f}ms norm {1e3*(t4-t3):.1f}ms eq={eq}")
print("mismatches", bad, "of", len(items))
