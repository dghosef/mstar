"""Vision embed/merger: eager bf16, compiled dynamic=True (head), compiled default (main-like) vs an fp32 eager reference."""
import sys, torch
sys.path.insert(0, "/root/guest-qwen35vl/rustext"); sys.path.insert(1, "/root/guest-qwen35vl/vhead")
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_5.components.vision import vision_interpolation, vision_position_ids
m = Qwen3_5DenseModel("Qwen/Qwen3.5-0.8B")
sub = m.get_submodule("vision_encoder", "cuda")
tower = sub.model
cfg = sub.config
grid = [(1, 24, 32), (1, 32, 24)]
n = sum(t * h * w for t, h, w in grid)
torch.manual_seed(0)
pv = torch.randn(n, 1536, device="cuda")
idx, wts = vision_interpolation(grid, cfg.num_grid_per_side, 2, torch.device("cuda"))
pos = vision_position_ids(grid, 2, torch.device("cuda"))
hid = torch.randn(n, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    ref_e = [t.float() for t in tower.float().embed(pv, idx, wts, pos)]; ref_m = tower.merger(hid.float()).float()
    tower.to(torch.bfloat16)
    def err(a, b): return ((a.float() - b).norm() / b.norm()).item()
    for name, fe, fm in (("eager", tower.embed, tower.merger),
                         ("compiled dynamic=True (head)", torch.compile(tower.embed, dynamic=True), torch.compile(tower.merger, dynamic=True)),
                         ("compiled default", torch.compile(tower.embed), torch.compile(tower.merger))):
        e = fe(pv, idx, wts, pos); mm = fm(hid)
        print(f"{name}: embed rel err {err(e[0], ref_e[0]):.2e}, merger rel err {err(mm, ref_m):.2e}")
