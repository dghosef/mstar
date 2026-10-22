"""CPU repro: an opaque RaggedAttentionCallable constructed lazily INSIDE a torch.compile'd region."""
import torch
from mstar.engine.resources import convenience as cv

class FakeAttn:
    def run(self, q, k, v, label):
        return q * 2 + v

class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fns = {}
    def fn_for(self, res):
        fn = self.fns.get(id(res))
        if fn is None:
            fn = self.fns[id(res)] = cv.RaggedAttentionCallable(res, "main", opaque=True)
        return fn
    def forward(self, x, res):
        return self.fn_for(res)(x, x, x) + 1

m = Mod()
f = torch.compile(m, dynamic=True)
resources = [FakeAttn(), FakeAttn()]  # e.g. double-buffered resource objects
x = torch.randn(4, 8)
for i in range(4):
    try:
        y = f(x, resources[i % 2])
        print(i, "ok", torch.allclose(y, x * 3 + 1), "registered", len(cv._RAGGED_CALLABLES),
              [fn._handle for fn in m.fns.values()])
    except Exception as e:
        print(i, "FAIL", type(e).__name__, str(e)[:300], "registered", len(cv._RAGGED_CALLABLES))
