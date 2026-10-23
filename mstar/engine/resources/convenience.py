import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.ragged.base import (
    RaggedAttnManager,
    RaggedBlockCausalAttnManager,
    RaggedCrossAttnManager,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.recurrent.pool import RecurrentStatePool


class AttentionCallable:
    """A convenience wrapper around kv and attn that wraps the KV write and
    attention call in one pure-tensor function, with helper methods for
    setting layer and label information.

    Must be used for  ``ulysses_attention``, which expects such a pure tensor
    callable. Recommended to use instance per transformer, *not* one per layer,
    as Dynamo specializes ``ulysses_attention`` on the identity of its
    `run_attention` argument, so a per-layer callable retraces that frame once
    per layer and blows the recompile limit.

    That sharing is why the label is one cursor for the whole stack. No model
    varies its label per layer today; one that needs to should thread the label
    explicitly rather than use this.

    For a cacheless attention, see :class:`RaggedAttentionCallable`.
    """

    def __init__(self, kv: KVManager, attn: AttentionManager | None=None):
        self.kv = kv
        # Can be changed at runtime, e.g., for a model that switches
        self.attn = attn

    @torch.compiler.disable
    def bind_step(self, label: str, attn: AttentionManager | None = None) -> None:
        if attn is not None:
            self.attn = attn
        assert self.attn is not None, (
            "no attention resource: pass `attn` here or at construction"
        )
        self.attn.set_default_label(label)
        self.kv.set_default_label(label)

    @property
    def label(self) -> str:
        """This step's label. Read through to the resource, not stored here, so
        one instance can drive a whole stack of per-layer callables — and so a
        layer that no longer takes a label as an argument can still reach it
        (the position resource carries no cursor, so `apply_qk` is passed this).
        """
        return self.kv.default_label

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        self.attn.set_default_layer_idx(layer_idx)
        self.kv.set_default_layer_idx(layer_idx)

    @torch.compiler.disable
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        if self.attn.requires_kv_write:
            self.kv.write_kv(k, v)
        return self.attn.run(q, kv_cache_layer=self.kv.layer_view(), k=k, v=v)


class LinearAttnCallable:
    """A recurrent state pool plus the resource running kernels against it,
    mirroring :class:`AttentionCallable`.

    The layer calls ``conv`` then the instance; this hands the pool's per-layer
    blocks over as plain tensors, so neither the layer nor the manager holds
    the pool.
    """

    def __init__(self, pool: RecurrentStatePool, attn: LinearAttnManager | None = None):
        self.pool = pool
        self.attn = attn
        self._layer_idx = 0

    @torch.compiler.disable
    def bind_step(self, label: str, attn: LinearAttnManager | None = None) -> None:
        if attn is not None:
            self.attn = attn
        assert self.attn is not None, (
            "no linear attention resource: pass `attn` here or at construction"
        )
        self.attn.set_default_label(label)

    @property
    def label(self) -> str:
        return self.attn.default_label

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        """The layer's index among the *recurrent* layers, not the stack's."""
        self._layer_idx = layer_idx
        self.attn.set_default_layer_idx(layer_idx)

    @torch.compiler.disable
    def conv(
        self, x: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor | None = None, activation: str | None = "silu",
    ) -> torch.Tensor:
        return self.attn.run_conv(
            x,
            conv_layer=self.pool.block("conv", self._layer_idx),
            weight=weight,
            bias=bias,
            activation=activation,
        )

    @torch.compiler.disable
    def __call__(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        a: torch.Tensor, b: torch.Tensor,
        a_log: torch.Tensor, dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn.run(
            q, k, v, a, b,
            state_layer=self.pool.block("state", self._layer_idx),
            a_log=a_log,
            dt_bias=dt_bias,
        )


class RaggedAttentionCallable:
    """`AttentionCallable` for a cacheless attention: ``(q, k, v) -> out`` over
    one declared span of a `RaggedAttnManager`.

    No ``kv`` and no layer cursor, because there is neither to carry: a ragged
    plan is keyed by the segment label, and the wrapper behind it is per
    (bucket, slot, label) rather than per layer. The label is therefore bound
    here at construction instead of being a cursor on the resource -- which is
    what lets one stack attend over several spans at once (a refiner over the
    image tokens next to a pass over all of them), the case `AttentionCallable`
    says to thread explicitly. A cross-attention pair of a
    ``RaggedCrossAttentionSpec`` resource is bound the same way, under
    ``cross_label(q_label, kv_label)``; its ``k`` / ``v`` are then the key span's
    tokens, packed in the same request order as ``q``. A
    ``RaggedBlockCausalAttentionSpec`` resource binds like a self-attention one.

    One instance per label, held for the life of the resource binding: a
    compiled transformer region guards on the identity of the callables it is
    handed, so handing it a fresh one per step retraces that frame every step
    until Dynamo gives up.
    """

    def __init__(
        self,
        attn: RaggedAttnManager | RaggedCrossAttnManager | RaggedBlockCausalAttnManager,
        label: str,
        opaque: bool = False,
    ):
        self.attn = attn
        self.label = label
        # ``opaque``: call through the ``mstar::ragged_attention`` custom op, which
        # a compiled region keeps in its graph as one node. Otherwise the call is a
        # graph break per layer, which also stops Inductor fusing the pointwise
        # work on either side of it: on Wan2.2's compiled, uncaptured DiT the
        # breaks cost ~85 ms of a ~1 s step in extra elementwise kernels.
        self._handle = _register_ragged(self) if opaque else None

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        if self._handle is not None:
            return _ragged_attention_op(q, k, v, self._handle)
        return self._run(q, k, v)

    @torch.compiler.disable
    def _run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.attn.run(q, k, v, label=self.label)


# Opaque callables by handle: a custom op takes tensors and scalars only, so it
# names its callable by index. One entry per callable built, i.e. per resource
# binding and label, which is a handful per process.
_RAGGED_CALLABLES: list[RaggedAttentionCallable] = []


# Eager even when the callable is built inside a compiled region (a node that
# creates its callables lazily on first forward): traced, the append is not
# replayed but the handle Dynamo read is still baked into the graph.
@torch.compiler.disable
def _register_ragged(fn: RaggedAttentionCallable) -> int:
    _RAGGED_CALLABLES.append(fn)
    return len(_RAGGED_CALLABLES) - 1


@torch.library.custom_op("mstar::ragged_attention", mutates_args=())
def _ragged_attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, handle: int) -> torch.Tensor:
    # the wrapper returns a fresh tensor (or a clone of its graph buffer), never an alias
    return _RAGGED_CALLABLES[handle]._run(q, k, v).to(q.dtype)


@_ragged_attention_op.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, handle: int) -> torch.Tensor:
    return torch.empty_like(q)


class Mamba2Callable:
    """``LinearAttnCallable`` for a Mamba-2 layer: the conv, then the SSD
    recurrence, each handed this layer's block of the pool as a plain tensor.

    Same cursor protocol (``bind_step`` once per stack, ``set_layer_idx`` per
    recurrent layer). The layer index counts recurrent layers only; a hybrid
    stack interleaves them with attention layers, and the pool is sized by its
    own count.
    """

    def __init__(self, pool: RecurrentStatePool, attn: LinearAttnManager | None = None):
        self.pool = pool
        self.attn = attn
        self._layer_idx = 0

    @torch.compiler.disable
    def bind_step(self, label: str, attn: LinearAttnManager | None = None) -> None:
        if attn is not None:
            self.attn = attn
        assert self.attn is not None, (
            "no Mamba-2 resource: pass `attn` here or at construction"
        )
        self.attn.set_default_label(label)

    @property
    def label(self) -> str:
        return self.attn.default_label

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int) -> None:
        self._layer_idx = layer_idx
        self.attn.set_default_layer_idx(layer_idx)

    @torch.compiler.disable
    def conv(
        self, x: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor | None = None, activation: str | None = "silu",
    ) -> torch.Tensor:
        return self.attn.run_conv(
            x,
            conv_layer=self.pool.block("conv", self._layer_idx),
            weight=weight,
            bias=bias,
            activation=activation,
        )

    @torch.compiler.disable
    def __call__(
        self,
        x: torch.Tensor, dt: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
        a_log: torch.Tensor, d: torch.Tensor | None, dt_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.attn.run(
            x, dt, b, c,
            ssm_layer=self.pool.block("ssm", self._layer_idx),
            a_log=a_log, d=d, dt_bias=dt_bias,
        )
