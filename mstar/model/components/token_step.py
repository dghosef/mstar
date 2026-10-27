"""The parts of a decoder node that samples one token per decode step which
don't depend on the model: the decode capture, the batch caps, feeding the
sampled token back in, and the stop checks."""
from __future__ import annotations

from functools import cached_property

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.model.submodule_base import ARNodeInputs, NodeInputs, TokenStop


class TokenStepNode:
    """Mixin for an ``ARNodeSubmodule`` whose forward returns ``new_token``.

    A subclass binds ``SAMPLER`` to its sampler resource's key and sets
    ``MAX_PREFILL_BATCH_SIZE`` and ``stop_token_ids``; it overrides the walk
    and loop names if its graph uses others. The stop checks are
    ``ARNodeSubmodule``'s, from the ``token_stop`` declared here.
    """

    DECODE_WALK = "decode"
    DECODE_LOOP = "decode_loop"
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    SAMPLER: str
    MAX_PREFILL_BATCH_SIZE: int
    stop_token_ids: frozenset[int]

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk=self.DECODE_WALK,
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
        ]

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        # the engine caps decode at its largest capture config, and with accelerator
        # graphs off leaves it uncapped; prefill isn't captured
        return None if graph_walk == self.DECODE_WALK else self.MAX_PREFILL_BATCH_SIZE

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # the decode loop feeds the sampled token back in as text_inputs
        if "new_token" in outputs:
            outputs["text_inputs"] = outputs["new_token"]

    @cached_property
    def token_stop(self) -> TokenStop:
        # read on first stop check, after ``__init__`` has set the stop ids
        return TokenStop(
            loop=self.DECODE_LOOP, buffer="new_token",
            stop_ids=frozenset(self.stop_token_ids), sampler=self.SAMPLER,
        )
