import cProfile, pstats, json, torch, sys
torch.set_num_threads(4)
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
m = Qwen3_5DenseModel("Qwen/Qwen3.5-0.8B", cache_dir=None)
items = json.load(open("/root/work/imgset/manifest.json"))
kind = sys.argv[1]
paths = [it["image"] for it in items if it["kind"] == kind]
def go():
    for p in paths * 3:
        img = m.load_image(p, "cpu").data
        m.process_prompt("What is this?", ["image", "text"], ["text"], tensors={"image_inputs": [img]})
go()
cProfile.run("go()", "/tmp/pp.prof")
pstats.Stats("/tmp/pp.prof").sort_stats("cumulative").print_stats(30)
