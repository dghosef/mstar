"""Stand-ins the Cosmos3 DiT tests share."""
from mstar.engine.resources import Segment, SubmoduleStep


def step_over(request_ids, *labels):
    """The part of a denoise step's declaration a batched forward reads: one
    segment per row and branch label."""
    return SubmoduleStep(segments=[Segment(rid, label, 1) for label in labels for rid in request_ids], steps={})


class StubSampler:
    """A request's sampler reduced to its schedule: ``update`` is the whole step
    and nothing rides the loop-back edges. Pair it with ``sub._solver_state = ()``."""

    def __init__(self, timesteps, update=lambda sample, velocity: sample - velocity):
        self.timesteps = timesteps
        self.update = update

    def initial_state(self, sample):
        return {}

    def step(self, model_output, sample, step_index, solver, generator=None):
        return self.update(sample, model_output), {}
