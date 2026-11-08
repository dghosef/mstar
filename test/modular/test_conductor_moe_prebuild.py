"""The conductor builds the fused-MoE align op before spawning workers only for a
model that asks: every other model paid the nvcc build, or hung on its lock."""
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from mstar.conductor.conductor import Conductor


@pytest.mark.parametrize("prebuild", [False, True])
def test_only_a_model_that_asks_builds_the_op(monkeypatch, prebuild):
    built = []
    align = types.ModuleType("mstar.utils.fused_moe.align")
    align._cuda_op_available = lambda: built.append(1)
    monkeypatch.setitem(sys.modules, "mstar.utils.fused_moe.align", align)
    monkeypatch.setattr("mstar.conductor.conductor.mp.get_context", lambda _method: SimpleNamespace(
        Process=lambda **_kw: SimpleNamespace(pid=1, start=lambda: None)))
    monkeypatch.setattr("mstar.conductor.conductor.atexit.register", lambda _fn: None)
    model = SimpleNamespace(prebuild_fused_moe=prebuild) if prebuild else object()
    conductor = mock.MagicMock(model=model, worker_ids=["worker_0"], _worker_devices=["cpu"], _mps_requested=False)
    Conductor._launch_workers(conductor)
    assert built == ([1] if prebuild else [])
