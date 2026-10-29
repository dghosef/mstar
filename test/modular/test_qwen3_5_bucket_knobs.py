"""The Qwen3.5 graph-bucket ladders: the decode one from the environment, the
prefill one from the deployment's model_kwargs."""
import importlib
from types import SimpleNamespace


def _reload(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    import mstar.model.qwen3_5.submodules as m
    return importlib.reload(m)


def test_prefill_ladder_from_model_kwargs():
    from mstar.model.qwen3_5.submodules import LLMSubmodule
    default = list(LLMSubmodule.PREFILL_TOKEN_BUCKETS)
    sub = LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)), prefill_token_buckets=[8192, 3072])
    assert sub.PREFILL_TOKEN_BUCKETS == [3072, 8192]
    # an instance override, not the class's
    assert LLMSubmodule.PREFILL_TOKEN_BUCKETS == default
    assert LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,))).PREFILL_TOKEN_BUCKETS == default


def test_decode_ladder_knob(monkeypatch):
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", "1,2,4,8,16,32")
    assert m.LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES == [1, 2, 4, 8, 16, 32]
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None)
    assert m.LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES[-1] == 128
