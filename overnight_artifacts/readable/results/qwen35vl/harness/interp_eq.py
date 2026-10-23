import torch, sys
sys.path.insert(0, "/root/work/mstar")
from mstar.model.qwen3_5.components.vision import vision_interpolation, vision_position_ids
from mstar.model.qwen3_5.components.rope import vision_position_ids as rope_vpos, text_position_ids
for grid in ([(1, 24, 32)], [(1, 72, 96), (1, 32, 24)], [(1, 128, 128)], [(2, 30, 46)]):
    a = vision_interpolation(grid, 48, 2, torch.device("cpu"))
    b = vision_interpolation(grid, 48, 2, torch.device("cuda"))
    p1 = vision_position_ids(grid, 2, torch.device("cpu")); p2 = vision_position_ids(grid, 2, torch.device("cuda"))
    g = torch.tensor(grid[0])
    r1 = rope_vpos(g, 2, 7.0); r2 = rope_vpos(g.cuda(), 2, 7.0, torch.device("cuda"))
    print(grid, torch.equal(a[0], b[0].cpu()), torch.equal(a[1], b[1].cpu()), torch.equal(p1, p2.cpu()), torch.equal(r1, r2.cpu()))
