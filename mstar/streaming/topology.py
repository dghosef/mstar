from dataclasses import dataclass, field
from typing import Callable

from mstar.graph.base import GraphEdge
from mstar.streaming.chunk_policy import ChunkPolicy


@dataclass
class StreamingGraphEdge(GraphEdge):
    """A graph edge that carries streaming data between partitions.

    Routed like a normal GraphEdge (producer is unaware it's streaming).
    On the consumer worker, the arriving tensors are buffered in a
    StreamBuffer and gated by a ChunkPolicy before satisfying the
    consuming node's input.
    """
    target_partition: str = ""

    def __post_init__(self):
        self.is_streaming = True


@dataclass(frozen=True)
class ChunkBatchHold:
    """How long a consumer may hold ready chunks back to batch them.

    For a consumer whose step costs far more than each extra row (a
    streaming vocoder): a ready chunk waits until ``min_chunks`` chunks of
    the connection are ready across requests, or the oldest has waited
    ``max_wait_s``. A stream's first chunk (time to first output), a chunk
    after the producer is done (end of the request) and any chunk while the
    worker has nothing in flight are never held.
    """
    max_wait_s: float
    min_chunks: int


@dataclass
class Connection:
    """Defines a streaming connection between two partitions."""
    from_partition: str
    to_partition: str
    edge_name: str
    chunk_policy_factory: Callable[[], ChunkPolicy]
    # None: hand each chunk to the consumer as soon as it is ready
    batch_hold: ChunkBatchHold | None = None


@dataclass
class PartitionTopology:
    """Declares how a model's computation is split into async partitions.

    Each partition has its own set of graph walks. Connections define
    streaming data flow between partitions via StreamBuffers.
    """
    partitions: list[str]
    connections: list[Connection] = field(default_factory=list)
