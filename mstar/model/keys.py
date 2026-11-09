"""Resource keys and walk names the shared model scaffolds read.

A model that builds on `SampledTokenDecoder` or `PackedBlockLoopEncoder`
declares these resources under these keys. Kept free of imports, so a model's
config module can name them without loading torch or the engine.

`Scope` is the prefix a multimodel gives a member (`asr.decode`, `llm.attn`) on the
walk, loop, resource-key and persisted-signal names the deployment shares.
"""
import dataclasses
import functools
from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    """A prefix on a member's shared names; the empty one changes nothing, so a
    model served alone keeps its names."""
    prefix: str = ""

    def __call__(self, name: str) -> str:
        return self.prefix + name

    def of(self, names):
        """``names``, a frozen dataclass of str defaults, with each prefixed."""
        return _scoped(self, names)


@functools.cache
def _scoped(scope: Scope, names):
    return dataclasses.replace(names, **{f.name: scope(getattr(names, f.name)) for f in dataclasses.fields(names)})


KV_CACHE = "kv_cache"
ROPE = "rope"
SAMPLER = "sampler"
# a vision tower's cacheless attention
VISION_ATTN = "vision_attn"
# text prefill and decode rows batched in one step (`Model.get_combined_graph_walks`)
MIXED = "mixed"
