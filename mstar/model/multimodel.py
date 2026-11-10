"""Several registered models served as one model: a multimodel.

Each member keeps its own nodes, walks, loops, resources and walk plan; the multimodel
only joins their declarations, so the conductor, validation and workers see one ordinary
model. A member's partitions are renamed ``<member>.<partition>``, and each member's plan
runs in its own partitions. Weights load per node, by the member that owns the node.

Names are shared deployment-wide (the conductor's per-request tables, the serving YAML's
``resources:``), so two members must not declare the same walk, loop, node, resource key or
signal name: building the multimodel refuses a clash, naming both members. A model that
joins gives its names to itself (Whisper's carry ``asr_``); two members of one family would
need the names to be prefixable, which is not built.

Members are ``Model`` instances, or ``{"model": <registered name>, **kwargs}`` specs the
multimodel builds, so a registered multimodel could come from the serving YAML's
``model_kwargs: {members: {...}}``; none is registered yet.

The first member takes a request's input; the output chunks are postprocessed by the member whose graph
emits their signal (`Model.postprocessor`).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

from mstar.conductor.request_info import PartitionDefinition
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphSection
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import Model
from mstar.streaming.topology import CLIENT, PartitionTopology


def _member(name: str, spec: Model | Mapping, cache_dir: str | None) -> Model:
    if isinstance(spec, Model):
        return spec
    from mstar.model.registry import get_model_class, model_init_kwargs

    kwargs = dict(spec)
    model = kwargs.pop("model")
    return get_model_class(model)(cache_dir=cache_dir, **{**model_init_kwargs(model), **kwargs})


def _signals(graphs: Iterable[GraphSection]) -> set[str]:
    """Every signal name a member's walks carry: node inputs and outputs, loops' outputs."""
    names: set[str] = set()
    for graph in graphs:
        for node in graph.get_nodes().values():
            names |= set(node.input_names) | {edge.name for edge in node.outputs}
        for loop in graph.get_loops().values():
            names |= {edge.name for edge in loop.outputs + loop.accumulated_outputs}
    return names


def refuse_clashes(kind: str, named: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """name -> the member that declares it; ValueError naming every name two members declare."""
    owner: dict[str, str] = {}
    clashes = []
    for member, names in named.items():
        for name in names:
            if owner.setdefault(name, member) != member:
                clashes.append(f"{name!r} ({owner[name]}, {member})")
    if clashes:
        raise ValueError(
            f"multimodel members share {kind} names: {', '.join(sorted(set(clashes)))}. "
            "A name is deployment-wide, so a member whose names are the shared defaults "
            "(a second model on the same scaffold) cannot join another."
        )
    return owner


class MultiModel(Model):

    def __init__(
        self, members: Mapping[str, Model | Mapping], cache_dir: str | None = None, model_path_hf: str = "",
    ):
        del model_path_hf  # each member names its own checkpoint
        if not members:
            raise ValueError("a multimodel needs at least one member")
        self.members = {name: _member(name, spec, cache_dir) for name, spec in members.items()}
        # the typed request's member: the first
        self.entry = next(iter(self.members.values()))
        for attr in ("PREPROCESS_ONLY_KWARGS", "INTERNAL_KWARGS", "LENGTH_KWARGS",
                     "PREPROCESS_TORCH_THREADS", "PREPROCESS_PARALLEL"):
            setattr(self, attr, getattr(self.entry, attr))
        self.INTERNAL_KWARGS = tuple(dict.fromkeys(k for m in self.members.values() for k in m.INTERNAL_KWARGS))

        walks = {name: m.get_graph_walk_graphs() for name, m in self.members.items()}
        refuse_clashes("walk", walks)
        self._node_owner = refuse_clashes("node", {
            name: {node for g in graphs.values() for node in g.get_nodes()} for name, graphs in walks.items()
        })
        refuse_clashes("loop", {
            name: {loop for g in graphs.values() for loop in g.get_loops()} for name, graphs in walks.items()
        })
        refuse_clashes("signal", {name: _signals(graphs.values()) for name, graphs in walks.items()})
        refuse_clashes("resource", {
            name: [spec.resource_key for spec in m.get_node_resources()] for name, m in self.members.items()
        })
        # multimodel partition -> (member, the member's own name for it)
        self._partition_of = {
            f"{name}.{p.name}": (name, p.name) for name, m in self.members.items() for p in m.get_partitions()
        }
        # output signal -> (member, modality), from the client edges of the members' walks
        self._emitted: dict[str, tuple[str, str]] = {}
        for name, graphs in walks.items():
            for graph in graphs.values():
                for node in graph.get_nodes().values():
                    for edge in node.outputs:
                        if edge.next_node == EMIT_TO_CLIENT:
                            self._emitted[edge.name] = (name, edge.output_modality)

    # -- the merged declarations ------------------------------------------------

    def _renamed(self, member: str, partition: str) -> str:
        return partition if partition == CLIENT else f"{member}.{partition}"

    def _merged(self, method: str, *args, **kwargs) -> dict:
        """The members' dicts as one; a key two members share is a clash, refused."""
        parts = {name: getattr(m, method)(*args, **kwargs) for name, m in self.members.items()}
        refuse_clashes(method, parts)
        return {key: value for part in parts.values() for key, value in part.items()}

    def get_graph_walk_graphs(self):
        return self._merged("get_graph_walk_graphs")

    def get_partitions(self) -> list[PartitionDefinition]:
        return [
            dataclasses.replace(p, name=self._renamed(name, p.name))
            for name, m in self.members.items() for p in m.get_partitions()
        ]

    def get_partition_topology(self) -> PartitionTopology:
        partitions, connections = [], []
        for name, m in self.members.items():
            topology = m.get_partition_topology()
            partitions += [self._renamed(name, p) for p in topology.partitions]
            connections += [dataclasses.replace(
                c, from_partition=self._renamed(name, c.from_partition),
                to_partition=self._renamed(name, c.to_partition),
            ) for c in topology.connections]
        return PartitionTopology(partitions=partitions, connections=connections)

    def get_node_resources(self):
        return [spec for m in self.members.values() for spec in m.get_node_resources()]

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

    def get_loop_caps(self, **model_kwargs):
        return self._merged("get_loop_caps", **model_kwargs)

    def get_request_resource_configs(self, input_modalities, output_modalities, model_kwargs):
        return self._merged("get_request_resource_configs", input_modalities, output_modalities, model_kwargs)

    def request_kwargs(self):
        kwargs = [m.request_kwargs() for m in self.members.values()]
        return None if any(k is None for k in kwargs) else frozenset().union(*kwargs)

    # -- a request: each member's plan runs in its own partitions ----------------

    def walk_plan(self, partition_name, input_modalities, output_modalities, input_signals, model_kwargs):
        member, own = self._partition_of[partition_name]
        return self.members[member].walk_plan(own, input_modalities, output_modalities, input_signals, model_kwargs)

    # -- delegated by node -------------------------------------------------------

    def _owner(self, node: str) -> Model:
        return self.members[self._node_owner[node]]

    def get_submodule(self, node_name, *args, **kwargs):
        return self._owner(node_name).get_submodule(node_name, *args, **kwargs)

    def get_sampling_config(self, node_name, model_kwargs=None):
        return self._owner(node_name).get_sampling_config(node_name, model_kwargs)

    def preprocess_fingerprint(self) -> str:
        return "+".join(f"{name}:{m.preprocess_fingerprint()}" for name, m in self.members.items())

    def get_autocast_dtype(self):
        return self.entry.get_autocast_dtype()

    def warmup_preprocess(self):
        for member in self.members.values():
            member.warmup_preprocess()

    # -- input -------------------------------------------------------------------

    def process_prompt(self, *args, **kwargs):
        return self.entry.process_prompt(*args, **kwargs)

    def process_stream_chunk(self, chunk):
        return self.entry.process_stream_chunk(chunk)

    def max_input_files(self):
        caps: dict[str, list[int | None]] = {}
        listed = [m.max_input_files() for m in self.members.values()]
        for modality in {mod for each in listed for mod in each}:
            caps[modality] = [each.get(modality) for each in listed]
        merged = {}
        for modality, values in caps.items():
            if any(v is not None and v > 0 for v in values):
                merged[modality] = max(v for v in values if v is not None)
            elif any(v is None for v in values):
                continue
            else:
                merged[modality] = 0
        return merged

    def _taking(self, modality: str) -> Model:
        return next((m for m in self.members.values() if m.max_input_files().get(modality, 0) > 0), self.entry)

    def load_image(self, *args, **kwargs):
        return self.entry.load_image(*args, **kwargs)

    def load_decoded_image(self, *args, **kwargs):
        return self.entry.load_decoded_image(*args, **kwargs)

    def load_audio(self, *args, **kwargs):
        return self._taking("audio").load_audio(*args, **kwargs)

    def load_video(self, *args, **kwargs):
        return self.entry.load_video(*args, **kwargs)

    # -- output ------------------------------------------------------------------

    def postprocessor(self, signal: str) -> Model:
        member = self._emitted.get(signal)
        return self if member is None else self.members[member[0]]

    def _emitting(self, modality: str) -> Model:
        names = [name for name, mod in self._emitted.values() if mod == modality]
        return self.members[names[-1]] if names else self.entry

    def get_max_output_tokens(self, **model_kwargs):
        return self.entry.get_max_output_tokens(**model_kwargs)

    def text_output(self):
        return self.entry.text_output()

    def postprocess(self, output, modality, request_kwargs=None):
        return self._emitting(modality).postprocess(output, modality, request_kwargs)

    def postprocess_inline(self, values, modality, request_kwargs):
        return self._emitting(modality).postprocess_inline(values, modality, request_kwargs)

    def get_output_sample_rate(self, modality="audio"):
        return self._emitting(modality).get_output_sample_rate(modality)

    def get_output_audio_channels(self, modality="audio"):
        return self._emitting(modality).get_output_audio_channels(modality)

    def get_output_frame_rate(self, modality, *args, **kwargs):
        return self._emitting(modality).get_output_frame_rate(modality, *args, **kwargs)

    def get_voices(self):
        return self._emitting("audio").get_voices()

    def get_default_voice(self):
        return self._emitting("audio").get_default_voice()
