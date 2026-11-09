"""Several registered models served as one model: a multimodel.

Each member keeps its own nodes, walks, loops, resources and walk transitions;
the multimodel only joins them. A ``Link`` streams one member node's output into
another member's node, as a streaming edge between their partitions, so one
request runs every member it reaches: one lifecycle and one abort, and each node
batches it with other requests as usual. Each member is placed by
``node_groups`` like any other node, so members can share a GPU or not.

Members are ``Model`` instances, or ``{"model": <registered name>, **kwargs}``
specs the multimodel builds, so a registered multimodel can come from YAML
``model_kwargs`` (``members``, ``links``). Registering one (registry,
``DEFAULT_CONFIGS``, a docs row) waits for the first real pair.

The first member takes the request's input (``process_prompt`` and the media
loaders), the last one formats its output; ``entry`` / ``output`` name others.

Members' names must not collide: nodes, walks, loops, resource keys and
persisted signals are deployment-wide, so a collision is refused here, naming
it. A member spec with ``scope: true`` is built with the prefix ``<member>.``
on its walks, loops, resource keys and persisted signals (``Scope``; the model
takes it as its ``scope`` kwarg), which is how two models that share names can
be members. In a multimodel's YAML (``node_groups``, ``resources:``) those are
the prefixed names. A member's partitions are renamed ``<member>.<partition>``;
the member sees its own partition names whenever the multimodel calls it.
"""
from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from mstar.conductor.request_info import PartitionDefinition
from mstar.distributed.base import ShardingConfig
from mstar.model.base import Model
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import CLIENT, Connection, PartitionTopology, StreamingGraphEdge


@dataclass(frozen=True)
class Link:
    """``source`` node's ``signal`` output, streamed into ``target`` node's
    ``input`` (the same name by default), ``chunk`` items at a time."""
    source: str
    signal: str
    target: str
    input: str | None = None
    chunk: int = 1
    # whether the target goes on after the source's stream ends
    continue_after_done: bool = True
    # A whole-input join: no stream. The target's member stays dormant until the
    # source's partition is done, then opens with its own ``process_prompt`` on
    # the text of the source's persisted ``signal`` (the source member's
    # ``text_output``). The target's ``input`` is unused.
    open: bool = False

    @property
    def edge_name(self) -> str:
        return self.input or self.signal


def _member(name: str, spec: Model | Mapping, cache_dir: str | None) -> Model:
    if isinstance(spec, Model):
        return spec
    from mstar.model.registry import get_model_class, model_init_kwargs

    kwargs = dict(spec)
    model = kwargs.pop("model")
    if kwargs.pop("scope", False):
        kwargs["scope"] = f"{name}."
    return get_model_class(model)(cache_dir=cache_dir, **{**model_init_kwargs(model), **kwargs})


def _owners(kind: str, named: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """name -> the member that declares it; ValueError naming every collision."""
    owner: dict[str, str] = {}
    clashes = []
    for member, names in named.items():
        for name in names:
            if owner.setdefault(name, member) != member:
                clashes.append(f"{name!r} ({owner[name]}, {member})")
    if clashes:
        raise ValueError(f"multimodel members share {kind} names: {', '.join(sorted(set(clashes)))}")
    return owner


class MultiModel(Model):

    def __init__(
        self, members: Mapping[str, Model | Mapping], links: Sequence[Link | Mapping] = (),
        entry: str | None = None, output: str | None = None,
        cache_dir: str | None = None, model_path_hf: str = "",
    ):
        del model_path_hf  # each member names its own checkpoint
        if not members:
            raise ValueError("a multimodel needs at least one member")
        self.members = {name: _member(name, spec, cache_dir) for name, spec in members.items()}
        self.links = tuple(link if isinstance(link, Link) else Link(**link) for link in links)
        names = list(self.members)
        self.entry = self.members[entry or names[0]]
        self.output = self.members[output or names[-1]]
        for attr in ("PREPROCESS_ONLY_KWARGS", "INTERNAL_KWARGS", "LENGTH_KWARGS",
                     "PREPROCESS_TORCH_THREADS", "PREPROCESS_PARALLEL"):
            setattr(self, attr, getattr(self.entry, attr))

        walks = {name: m.get_graph_walk_graphs() for name, m in self.members.items()}
        _owners("walk", walks)
        self._node_owner = _owners("node", {
            name: {node for g in graphs.values() for node in g.get_nodes()} for name, graphs in walks.items()
        })
        _owners("loop", {name: {lp for g in graphs.values() for lp in g.get_loops()} for name, graphs in walks.items()})
        _owners("resource", {name: [s.resource_key for s in m.get_node_resources() or []]
                             for name, m in self.members.items()})
        self._persisted = {
            name: {e.name for g in graphs.values() for n in g.get_nodes().values() for e in n.outputs if e.persist}
            for name, graphs in walks.items()
        }
        _owners("persisted signal", self._persisted)
        # multimodel partition -> (member, the member's own name for it)
        self._partition_of = {
            f"{name}.{p.name}": (name, p.name) for name, m in self.members.items() for p in m.get_partitions()
        }
        for link in self.links:
            for node in (link.source, link.target):
                if node not in self._node_owner:
                    raise ValueError(f"multimodel link names node {node!r}, which no member has")
            if self._node_owner[link.source] == self._node_owner[link.target]:
                raise ValueError(f"multimodel link {link.source!r} -> {link.target!r} stays inside one member")
            if link.open and link.signal not in self._persisted[self._node_owner[link.source]]:
                raise ValueError(
                    f"open link {link.source!r} -> {link.target!r}: {link.signal!r} is not a signal "
                    f"{self._node_owner[link.source]!r} persists"
                )
        opened = [link.target for link in self.links if link.open]
        if len({self._node_owner[node] for node in opened}) != len(opened):
            raise ValueError("a member can be opened by one link only")

    # -- names -------------------------------------------------------------

    def _owner(self, node: str) -> Model:
        return self.members[self._node_owner[node]]

    def _renamed(self, member: str, partition: str) -> str:
        return partition if partition == CLIENT else f"{member}.{partition}"

    def _partition_of_node(self, node: str) -> str:
        member = self._node_owner[node]
        walks = self.members[member].get_graph_walk_graphs()
        for p in self.members[member].get_partitions():
            if any(node in walks[w].get_nodes() for w in p.graph_walks if w in walks):
                return self._renamed(member, p.name)
        raise ValueError(f"node {node!r} is in no partition of member {member!r}")

    def _merged(self, method: str, *args, **kwargs) -> dict:
        out: dict = {}
        for member in self.members.values():
            out.update(getattr(member, method)(*args, **kwargs))
        return out

    def _streams(self) -> list[Link]:
        return [link for link in self.links if not link.open]

    # -- the graph -----------------------------------------------------------

    def get_graph_walk_graphs(self):
        graphs = self._merged("get_graph_walk_graphs")
        for link in self._streams():
            target = self._partition_of_node(link.target)
            for walk, graph in graphs.items():
                if link.source not in graph.get_nodes():
                    continue
                graphs[walk] = graph = copy.deepcopy(graph)
                graph.get_nodes()[link.source].outputs.append(StreamingGraphEdge(
                    next_node=link.target, name=link.edge_name, target_partition=target,
                ))
        return graphs

    def get_partitions(self) -> list[PartitionDefinition]:
        producers = {}
        opened = {self._node_owner[link.target]: link for link in self.links if link.open}
        for link in self._streams():
            producers.setdefault(self._partition_of_node(link.target), []).append(
                self._partition_of_node(link.source))
        out = []
        for name, member in self.members.items():
            for p in member.get_partitions():
                renamed = self._renamed(name, p.name)
                link = opened.get(name)
                out.append(dataclasses.replace(
                    p, name=renamed,
                    producer_partitions=[self._renamed(name, q) for q in p.producer_partitions]
                    + producers.get(renamed, []),
                    open_after=None if link is None else self._partition_of_node(link.source),
                    open_signal="" if link is None else link.signal,
                ))
        return out

    def get_partition_topology(self) -> PartitionTopology:
        partitions, connections = [], []
        for name, member in self.members.items():
            topology = member.get_partition_topology()
            partitions += [self._renamed(name, p) for p in topology.partitions]
            connections += [dataclasses.replace(
                c, from_partition=self._renamed(name, c.from_partition),
                to_partition=self._renamed(name, c.to_partition),
            ) for c in topology.connections]
        for link in self._streams():
            connections.append(Connection(
                from_partition=self._partition_of_node(link.source),
                to_partition=self._partition_of_node(link.target),
                edge_name=link.edge_name,
                chunk_policy_factory=lambda link=link: FixedChunkPolicy(
                    chunk_size=link.chunk, continue_after_done=link.continue_after_done),
            ))
        return PartitionTopology(partitions=partitions, connections=connections)

    def get_node_resources(self):
        return [spec for m in self.members.values() for spec in m.get_node_resources() or []]

    def get_lora_specs(self):
        return [spec for m in self.members.values() for spec in m.get_lora_specs()]

    def get_combined_graph_walks(self):
        return self._merged("get_combined_graph_walks")

    def get_chunking_policies(self):
        return self._merged("get_chunking_policies")

    def rebuild_policies(self):
        return self._merged("rebuild_policies")

    def kv_labels(self):
        return self._merged("kv_labels")

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        for member in self.members.values():
            member.validate_config_yaml(config, config_path)

    def get_default_sharding_config(self) -> ShardingConfig:
        merged = ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})
        for config in (m.get_default_sharding_config() for m in self.members.values()):
            for field in dataclasses.fields(ShardingConfig):
                mine, theirs = getattr(merged, field.name), getattr(config, field.name)
                if isinstance(theirs, dict):
                    mine.update(theirs)
                elif isinstance(theirs, list):
                    mine += theirs
                else:  # the node sets
                    setattr(merged, field.name, set(mine) | set(theirs))
        return merged

    def get_submodule(self, node_name, *args, **kwargs):
        return self._owner(node_name).get_submodule(node_name, *args, **kwargs)

    def get_sampling_config(self, node_name, model_kwargs=None):
        return self._owner(node_name).get_sampling_config(node_name, model_kwargs)

    def preprocess_fingerprint(self) -> str:
        return "+".join(f"{name}:{m.preprocess_fingerprint()}" for name, m in self.members.items())

    def get_autocast_dtype(self):
        return self.entry.get_autocast_dtype()

    # -- a request -----------------------------------------------------------

    def get_initial_forward_pass_args(self, partition_name, *args, **kwargs):
        member, own = self._partition_of[partition_name]
        return self.members[member].get_initial_forward_pass_args(own, *args, **kwargs)

    def get_partition_forward_pass_args(
        self, partition_name, partition_metadata, persist_signals, incoming_connections=None,
    ):
        member, own = self._partition_of[partition_name]
        # a member sees no other member's persisted signals (the request's own
        # inputs are in the dict too, and stay)
        others = set().union(*(names for m, names in self._persisted.items() if m != member))
        persist_signals = {k: v for k, v in persist_signals.items() if k not in others}
        # the member reads its own partition names off its connections; a link's
        # producer, another member's partition, keeps the multimodel's name
        prefix = f"{member}."
        incoming = [dataclasses.replace(
            c, from_partition=c.from_partition.removeprefix(prefix),
            to_partition=c.to_partition.removeprefix(prefix),
        ) for c in incoming_connections or []]
        return self.members[member].get_partition_forward_pass_args(
            own, partition_metadata, persist_signals, incoming,
        )

    def get_request_resource_configs(self, partition_fwd_args, model_kwargs=None):
        out = {}
        for name, member in self.members.items():
            own = {self._partition_of[p][1]: args for p, args in partition_fwd_args.items()
                   if self._partition_of.get(p, ("",))[0] == name}
            out.update(member.get_request_resource_configs(own, model_kwargs))
        return out

    def get_loop_caps(self, **model_kwargs):
        return self._merged("get_loop_caps", **model_kwargs)

    def get_max_output_tokens(self, **model_kwargs):
        return self.output.get_max_output_tokens(**model_kwargs)

    def request_kwargs(self):
        kwargs = [m.request_kwargs() for m in self.members.values()]
        return None if any(k is None for k in kwargs) else frozenset().union(*kwargs)

    # -- a deferred member's input -----------------------------------------------

    def open_member(self, partition: str) -> Model:
        """The member that owns multimodel partition ``partition``."""
        return self.members[self._partition_of[partition][0]]

    def open_text(self, partition: str, ids) -> str:
        """The text of ``ids``, the persisted ``open_signal`` of the link that
        opens ``partition``'s member: the source member's own detokenizer,
        without its stop ids."""
        member = self._partition_of[partition][0]
        (link,) = [lk for lk in self.links if lk.open and self._node_owner[lk.target] == member]
        text = self._owner(link.source).text_output()
        if text is None:
            raise ValueError(f"member {self._node_owner[link.source]!r} declares no text output")
        kept = [i for i in ids.reshape(-1).tolist() if i not in text.stop_token_ids]
        return text.detokenizer.to_text_bytes(kept).decode("utf-8", errors="replace").strip()

    # -- input, from the entry member ------------------------------------------

    def process_prompt(self, *args, **kwargs):
        return self.entry.process_prompt(*args, **kwargs)

    def process_stream_chunk(self, chunk):
        return self.entry.process_stream_chunk(chunk)

    def max_input_files(self):
        return self.entry.max_input_files()

    def load_image(self, *args, **kwargs):
        return self.entry.load_image(*args, **kwargs)

    def load_decoded_image(self, *args, **kwargs):
        return self.entry.load_decoded_image(*args, **kwargs)

    def load_audio(self, *args, **kwargs):
        return self.entry.load_audio(*args, **kwargs)

    def load_video(self, *args, **kwargs):
        return self.entry.load_video(*args, **kwargs)

    def warmup_preprocess(self):
        return self.entry.warmup_preprocess()

    # -- output, from the output member ----------------------------------------

    def text_output(self):
        return self.output.text_output()

    def postprocess(self, *args, **kwargs):
        return self.output.postprocess(*args, **kwargs)

    def postprocess_inline(self, *args, **kwargs):
        return self.output.postprocess_inline(*args, **kwargs)

    def get_output_sample_rate(self, *args, **kwargs):
        return self.output.get_output_sample_rate(*args, **kwargs)

    def get_output_audio_channels(self, *args, **kwargs):
        return self.output.get_output_audio_channels(*args, **kwargs)

    def get_output_frame_rate(self, *args, **kwargs):
        return self.output.get_output_frame_rate(*args, **kwargs)

    def get_voices(self):
        return self.output.get_voices()

    def get_default_voice(self):
        return self.output.get_default_voice()
