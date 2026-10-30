"""Paged attention through FlashInfer's prefill/decode wrappers."""

from collections.abc import Mapping

import torch

from mstar.engine.resources.attn.base import (
    AttentionManager,
    AttentionWrapper,
    EagerSlotKey,
    WorkspacePool,
    captured_page_capacity,
)
from mstar.engine.resources.attn.config import AttentionStep, refuse_context_only
from mstar.engine.resources.attn.wrappers import (
    FlashInferContextParallelWrapper,
    FlashInferDecodeWrapper,
    FlashInferMixedWrapper,
    FlashInferPrefillWrapper,
    check_flashinfer_head_dim,
)
from mstar.engine.resources.base import CGSlotKey, Resource
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import SlotLease, StepContext
from mstar.utils.profiler import PhaseLaps


class FlashInferManager(AttentionManager):
    supports_context_parallel = True

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: PagedKVConfig,
        backend: str="auto",
        splits: tuple[CombinedSplit, ...] = (),
        max_rows_in_combined: Mapping[str, int | None] | None = None,
        decode_tensor_cores: bool = True,
        sliding_window: int | None = None,
    ):
        self._kv_cache_name = kv_cache
        self._splits = {split.combined_walk: split for split in splits}
        self._max_rows_in_combined = max_rows_in_combined or {}
        self._device = device
        self._dtype = dtype
        # FlashInfer counts preceding positions; M* counts the current token too.
        self._window_left = -1 if sliding_window is None else sliding_window - 1

        # label to plan state
        self._current_plan_states: dict[str, AttentionWrapper] = {}

        # persistent, because constructing one allocates FlashInfer's own
        # workspaces and is far from free per step
        self._eager_plan_states: dict[EagerSlotKey, AttentionWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        self._preplan_states: dict[str, AttentionWrapper] = {}
        self._preplanned = False

        self._kv_config = kv_config
        check_flashinfer_head_dim(kv_config.head_dim, device)     # at load, not at the first plan
        self._wrapper_kv_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            num_kv_heads=kv_config.num_kv_heads,
            head_dim=kv_config.head_dim,
            page_size=kv_config.page_size,
            max_num_pages=kv_config.max_num_pages,
            device=device,
            backend=backend
        )
        self._decode_tensor_cores = decode_tensor_cores

        self._workspaces = WorkspacePool(device)
        # the KV pool's placement (`bind_dependencies`)
        self._placement = None

    def depends_on(self):
        return {self._kv_cache_name}

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        placement = getattr(resources.get(self._kv_cache_name), "placement", None)
        self._placement = placement
        if placement is None:
            return
        # the wrappers read this rank's slots of each page; under TP placement each attends
        # them for every partner's query heads (`ContextParallel.exchange_queries`)
        cfg = self._kv_config
        self._wrapper_kv_kwargs["page_size"] = cfg.page_size // placement.size
        self._wrapper_kv_kwargs["num_qo_heads"] = cfg.num_qo_heads * (1 if placement.over_sp else placement.size)
        self._new_kwargs = dict(
            num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
            device=self._device, backend=self._wrapper_kv_kwargs["backend"], q_data_type=self._dtype,
        )

    def _cg_wrapper(
        self, lease: SlotLease, label: str, num_rows: int, force_prefill: bool,
    ) -> AttentionWrapper:
        """The captured-graph wrapper for one (bucket, slot, label).

        Built on the first plan for that key rather than up front: which labels
        a walk plans under is the step's to declare, and the first plan for a
        bucket runs during capture, before the graph is recorded, so the wrapper
        is just as persistent either way.

        ``num_rows`` is the qo_indptr row count this label actually plans —
        the FlashInfer batch_size for CUDA-graph mode, which must stay fixed
        for the wrapper's lifetime. It equals ``bucket.bs`` (one row per
        request) for an ordinary label, but a label that combines several
        source labels into one plan (e.g. cond+uncond batched CFG) always
        contributes several rows per request; using ``bucket.bs`` there
        under-allocates the wrapper's fixed buffers and every later plan()
        call overruns them with a "runtime batch size" mismatch.
        """
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_plan_states.get(key)
        if wrapper is not None:
            return wrapper

        bucket = lease.bucket
        buffer = self._workspaces.get(label, lease.slot)
        kv_kwargs = {
            **self._wrapper_kv_kwargs,
            "max_num_pages": captured_page_capacity(num_rows, bucket.num_tokens, self._kv_config),
        }
        split = self._splits.get(bucket.graph_walk)
        if self._placement is not None:
            # the mixed wrapper is not used: the context wrapper takes rows of any length
            wrapper = self._context_parallel_wrapper(
                label, lease.slot, bucket.bs == bucket.num_tokens and not force_prefill,
                dict(batch_size=num_rows, accelerator_graph=True, **kv_kwargs), bucket.num_tokens,
            )
        elif split is not None:
            # its own workspaces: both wrappers are planned for the one step
            wrapper = FlashInferMixedWrapper(
                decode_workspace=self._workspaces.get(f"{label}.split_decode", lease.slot),
                prefill_workspace=self._workspaces.get(f"{label}.split_prefill", lease.slot),
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                max_prefill_rows=split.max_prefill_rows(
                    self._max_rows_in_combined, num_rows,
                ),
                use_tensor_cores=self._decode_tensor_cores,
                **kv_kwargs,
            )
        elif bucket.bs == bucket.num_tokens and not force_prefill:
            wrapper = FlashInferDecodeWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                accelerator_graph=True,
                use_tensor_cores=self._decode_tensor_cores,
                **kv_kwargs,
            )
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                accelerator_graph=True,
                **kv_kwargs,
            )
        self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(
        self, label: str, is_decode: bool, slot: int
    ) -> AttentionWrapper:
        """The persistent eager wrapper for one (label, kind, slot).

        Takes the workspace this slot's captured wrappers already hold: the
        two never run at once, since the pipeline retires step N before the
        host stages N+2 and a slot repeats no sooner than that.
        """
        key = EagerSlotKey(label=label, slot=slot, is_decode=is_decode)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None and self._placement is not None:
            wrapper = self._eager_plan_states[key] = self._context_parallel_wrapper(
                label, slot, is_decode, dict(self._wrapper_kv_kwargs),
            )
        if wrapper is None:
            workspace = self._workspaces.get(label, slot)
            if is_decode:
                wrapper = FlashInferDecodeWrapper(
                    workspace_buffer=workspace,
                    use_tensor_cores=self._decode_tensor_cores,
                    **self._wrapper_kv_kwargs,
                )
            else:
                wrapper = FlashInferPrefillWrapper(workspace_buffer=workspace, **self._wrapper_kv_kwargs)
            self._eager_plan_states[key] = wrapper
        return wrapper

    def _context_parallel_wrapper(
        self, label: str, slot: int, decode: bool, kv_kwargs: dict, max_total_tokens: int | None = None,
    ) -> FlashInferContextParallelWrapper:
        """The wrappers a step attends a placed pool on: its context's on the label's workspace,
        and for a step that isn't one token a row, its own tokens' on a workspace of their own
        (both are planned for the one step)."""
        from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper

        workspace = self._workspaces.get(label, slot)
        if decode:
            # under TP placement a rank attends every partner's query heads over one KV head, a GQA
            # group FlashInfer's CUDA-core decode may have no kernel for
            tensor_cores = self._decode_tensor_cores or not self._placement.over_sp
            context = FlashInferDecodeWrapper(workspace_buffer=workspace, use_tensor_cores=tensor_cores, **kv_kwargs)
            new = None
        else:
            context = FlashInferPrefillWrapper(
                workspace_buffer=workspace, max_total_tokens=max_total_tokens, **kv_kwargs,
            )
            new = FlashInferRaggedPrefillWrapper(
                self._workspaces.get(f"{label}.own_tokens", slot),
                max_num_segments=kv_kwargs.get("batch_size"), max_total_tokens=max_total_tokens,
                accelerator_graph=kv_kwargs.get("accelerator_graph", False), **self._new_kwargs,
            )
        # a captured wrapper's every query token: a row's at least, and the bucket's
        captured_tokens = max(kv_kwargs.get("batch_size") or 0, max_total_tokens or 0)
        return FlashInferContextParallelWrapper(context, new, self._device, captured_tokens)

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # FlashInfer's plan stages the schedule into a pinned buffer it holds
        # per wrapper and H2Ds it on the stream (scheduler.cuh's
        # cudaMemcpyAsync out of page_locked_int_workspace_buffer), so one
        # wrapper per slot is what keeps plan(N+1) off step N's queued copy.
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        if self._window_left >= 0 and not step.causal:
            raise ValueError("sliding_window requires causal attention")
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a captured graph step: an eager wrapper shares its "
            "workspace with the captured one on the same slot"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            # the wrappers were planned a step early against this same step's
            # KV plan; nothing left to do but promote them
            self._current_plan_states = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, f"Attention Manager expected plan result from {self._kv_cache_name}"

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states

        plan_states.clear()
        laps = PhaseLaps("res.pre_plan.attn." if ctx.is_preplan else "res.plan.attn.")
        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            if lease is not None:
                num_rows = indptrs.qo_indptr.shape[0] - 1
                wrapper = self._cg_wrapper(lease, label, num_rows, ctx.force_prefill)
            else:
                is_decode = not ctx.force_prefill and bool(
                    indptrs.qo_indptr[-1] == len(indptrs.qo_indptr) - 1
                )
                wrapper = self._eager_wrapper(label, is_decode, ctx.slot)

            if self._placement is not None:
                laps.lap("wrapper")
                wrapper.plan(self._placement, self._kv_config.page_size, kv_out, causal=step.causal, dtype=self._dtype)
                laps.lap("wrapper_plan")
                plan_states[label] = wrapper
                continue
            kwargs = indptrs.to_kwargs_dict()
            laps.lap("wrapper")
            # TODO: cache the latest plan state
            wrapper.plan(
                causal=step.causal, dtype=self._dtype, window_left=self._window_left,
                **kwargs,
            )
            laps.lap("wrapper_plan")
            plan_states[label] = wrapper

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    ### Submodule-level functionality
    def reads_step_kv(self, label: str | None = None) -> bool:
        # over a placed pool a step that isn't one token a row attends its own tokens from k and v
        if self._placement is None:
            return False
        return self._current_plan_states[label or self._default_label].new is not None

    @torch.compiler.disable
    def qo_indptr_buf(self, label: str="main") -> torch.Tensor | None:
        if label not in self._current_plan_states:
            return
        # decode wrappers never set one
        return getattr(self._current_plan_states[label], "_qo_indptr_buf", None)

    def select_last_hidden(
        self, hidden: torch.Tensor, label: str="main"
    ) -> torch.Tensor:
        """
        Select last token of the hidden vector per request, used for sampling
        from prefill.
        """
        qo_indptr_buf = self.qo_indptr_buf(label)
        last_token_indices = (qo_indptr_buf[1:] - 1).long()
        return hidden.index_select(0, last_token_indices)

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # The K/V is already in the pages (the layer wrote it through the KV
        # resource) and the planned wrapper carries the layout, so `k`/`v` are
        # read only over a placed pool, where a step's own tokens attend each
        # other from them; `layer_idx` is the dense backend's. Accepted so one
        # layer body serves either backend — see `requires_kv_write`.
        del layer_idx
        if label is None:
            label = self._default_label
        if self._placement is not None:
            return self._attend_placed(q, label, kv_cache_layer, k, v)
        return self._attend(q, label, kv_cache_layer)

    @torch.compiler.disable
    def _attend_placed(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
        k: torch.Tensor | None, v: torch.Tensor | None,
    ) -> torch.Tensor:
        """Every rank's partial over its slots, exchanged (the same collectives on every rank,
        every layer) and merged with this step's own tokens' attention when it has one."""
        wrapper: FlashInferContextParallelWrapper = self._current_plan_states[label]
        if wrapper.new is not None and (k is None or v is None):
            raise ValueError(
                "FlashInferManager over a context-parallel KV pool attends a step's own tokens from "
                "the k and v run is passed; this step has more than one token a row and got none"
            )
        return self._placement.attend(
            q, lambda queries: wrapper.attend_context(queries, kv_cache_layer),
            None if wrapper.new is None else wrapper.attend_new(q, k, v),
        ).to(q.dtype)

    @torch.compiler.disable
    def _attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        o = self._current_plan_states[label].run(q, kv_cache_layer)
        if o.dtype != q.dtype:
            o = o.to(q.dtype)
        return o
