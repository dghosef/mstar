"""The deployment's admission policy: the order requests are admitted and preempted in, and which ones are never
preempted. Serving config ``admission_policy``: ``fcfs`` (the default), ``priority``, or
``"package.module:Class"`` naming a subclass of `AdmissionPolicy`.

The conductor asks the policy once per request, when it takes the request in. The answers are facts of the request
from then on, the same on every worker and rank:

- ``priority`` places the request in one order, ahead of its arrival. The conductor's queue
  (``max_concurrent_requests``) and each pool's queue admit in that order, a returning request is tried in it, and
  a request that runs short of a pool can preempt only requests after it, the last first (`order`).
- ``unpausable`` makes it reserve its whole footprint in every pool and never be preempted. A pool also guarantees a
  request it couldn't take pages back from, whatever the policy says.

Every request finishes once no request ordered before it keeps arriving (docs/design/admission_and_preemption.md §5.5).
Under strict priority a stream of earlier-ordered requests can hold a later one back, and drop it again each time it
comes back, for as long as the stream lasts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mstar.utils.policy import resolve_policy

if TYPE_CHECKING:
    from mstar.utils.ipc_format import NewRequestConductor

# A priority is in [-PRIORITY_BOUND, PRIORITY_BOUND), and an arrival number below 2**ARRIVAL_BITS, so `order` stays
# below 2**59, under the bands the worker's line order adds (`micro_scheduler._AHEAD`).
PRIORITY_BOUND = 2**18
ARRIVAL_BITS = 40


def order(priority: int, arrival: int) -> int:
    """The request's place in the order every worker reads (`CurrentForwardPassInfo.arrival`): by ``priority``, lower
    first, then by the conductor's arrival number."""
    return ((priority + PRIORITY_BOUND) << ARRIVAL_BITS) + arrival


class AdmissionPolicy:
    """First come, first served. Every request has priority 0, and a streamed request with an output its client plays
    as it arrives (``unpausable_outputs``, by default the model's ``UNPAUSABLE_OUTPUTS``) is never preempted.

    A subclass overrides either method. Both must answer from the request alone, since the conductor asks once."""

    def __init__(self, unpausable_outputs: frozenset[str]) -> None:
        self.unpausable_outputs = unpausable_outputs

    def priority(self, body: NewRequestConductor) -> int:
        """The request's priority, lower first, in [-PRIORITY_BOUND, PRIORITY_BOUND). Raise ``ValueError`` to refuse
        the request with a 400."""
        return 0

    def unpausable(self, body: NewRequestConductor) -> bool:
        """Whether the request reserves its whole footprint and is never preempted."""
        return body.streaming and not self.unpausable_outputs.isdisjoint(body.initial_output_modalities)


class PriorityPolicy(AdmissionPolicy):
    """By the request's ``priority`` kwarg, an integer, lower first and 0 when unset, as vLLM orders them."""

    def priority(self, body: NewRequestConductor) -> int:
        priority = (body.model_kwargs or {}).get("priority", 0)
        if priority is None:
            return 0
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError(f"priority must be an integer, got {priority!r}.")
        if not -PRIORITY_BOUND <= priority < PRIORITY_BOUND:
            raise ValueError(f"priority must be in [{-PRIORITY_BOUND}, {PRIORITY_BOUND}), got {priority}.")
        return priority


ADMISSION_POLICIES: dict[str, type[AdmissionPolicy]] = {"fcfs": AdmissionPolicy, "priority": PriorityPolicy}


def admission_policy(config: dict, unpausable_outputs: frozenset[str]) -> AdmissionPolicy:
    """The policy the serving config names, built with the outputs that can't pause."""
    spec = config.get("admission_policy", "fcfs")
    return resolve_policy("admission_policy", spec, ADMISSION_POLICIES, AdmissionPolicy)(unpausable_outputs)
