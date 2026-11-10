"""What the conductor sends and releases at each step of a model's walk plan.
The preprocess worker persists every input signal until the conductor
unpersists it, so without the releases pixel values, audio features and prompt
tokens stay held for the whole request.

Drives the real ``Conductor.__init__`` and ``Conductor.run`` with worker launch
and ZMQ stubbed: the stub communicator records every send and replays one
scripted batch of inbound messages per run-loop iteration.
"""
from collections import deque
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

import mstar.conductor.conductor as conductor_mod  # noqa: E402
from mstar.conductor.conductor import Conductor, ForwardPassArgs, _walk_io  # noqa: E402
from mstar.conductor.request_info import (  # noqa: E402
    FINAL_PREFILL,
    PartitionDefinition,
    PartitionState,
    WalkStep,
)
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig  # noqa: E402
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION  # noqa: E402
from mstar.model.base import KVLabel, LabelLayout, Span  # noqa: E402
from mstar.model.submodule_base import ChunkingPolicy, chunk_tick_name  # noqa: E402
from mstar.streaming.chunk_policy import FixedChunkPolicy  # noqa: E402
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge  # noqa: E402
from mstar.testing_utils.model import StubModel  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    AbortRequest,
    ConductorMessage,
    ConductorMessageType,
    FailRequests,
    NewRequestConductor,
    ReadsDone,
    SetupDone,
    WorkerGraphsDone,
)

PREPROCESS = "api_server_preprocess_worker"


class _StubComm:
    def __init__(self):
        self.sent = []
        self.batches = deque()

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return self.batches.popleft() if self.batches else []

    def poll_for_messages(self, timeout_ms):
        return bool(self.batches)

    def flush(self):
        return False  # every send above went out


class _ScriptDoneError(Exception):
    pass


class _ToyModel(StubModel):
    """Walks, partitions (``{name: walks}``, one default partition when
    omitted), streams and each partition's plan come from the test."""

    def __init__(self, walks, plans, groups=None, connections=(), growth=False, chunked=()):
        self._walks, self._plans = walks, plans
        self._partitions = [
            PartitionDefinition(name=name, graph_walks=set(walks_)) for name, walks_ in groups.items()
        ] if groups else None
        self._connections, self._growth, self._chunked = list(connections), growth, chunked

    def get_node_resources(self):
        if not self._growth:
            return []
        return [KVSpec(resource_key="kv", nodes={"llm"}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=1024, max_num_pages=64, page_size=16,
        ))]

    def kv_labels(self):
        return {"kv": {"main": KVLabel(growth=(("decode", "decode_loop"),))}} if self._growth else {}

    def get_chunking_policies(self):
        return {"llm": (ChunkingPolicy(walks=self._chunked, chunkable=True, max_batch_tokens=64),)} \
            if self._chunked else {}

    def get_graph_walk_graphs(self):
        return self._walks

    def get_partitions(self):
        return self._partitions or super().get_partitions()

    def get_partition_topology(self):
        if not self._partitions:
            return super().get_partition_topology()
        return PartitionTopology(partitions=[p.name for p in self._partitions], connections=self._connections)

    def kv_layout(self, layout, length):
        return {}   # (a model that spawns has one: the conductor asks it for an agent's)

    def walk_plan(self, partition_name, input_modalities, output_modalities, input_signals, model_kwargs):
        plan = self._plans["agent" if model_kwargs.get("agent_child") else partition_name]
        return list(plan)


def _conductor(model, node_groups, tmp_path, monkeypatch, **config) -> Conductor:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"max_seq_len": 1024, "node_groups": node_groups, **config}))
    comm = _StubComm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    c = Conductor(model, str(path), socket_path_prefix=str(tmp_path / "s"))
    c.comm = comm
    return c


def _run(c: Conductor, batches):
    """Run the real ``Conductor.run`` over ``batches``, one per iteration."""
    setup = [_msg(ConductorMessageType.SETUP_DONE, SetupDone(w)) for w in c.worker_ids]
    c.comm.batches.extend([setup, *batches])

    def poll():
        if not c.comm.batches:
            raise _ScriptDoneError

    c._poll_worker_liveness = poll
    with pytest.raises(_ScriptDoneError):
        c.run()


def _msg(kind, body):
    return ConductorMessage(message_type=kind, body=body)


def _new_request(rid, signals):
    return _msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
        request_id=rid, initial_signals=signals, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={},
    ))


def _info(uuid, entity=PREPROCESS, node=None, walk=None):
    return TensorPointerInfo(
        dims=[4], dtype=torch.int64, nbytes=32, address=0, stride=[1], uuid=uuid,
        source_session_id="s", source_entity=entity, _source_node_name=node, _source_graph_walk=walk,
    )


def _sends(c, kinds=("NEW_REQUEST", "UNPERSIST_TENSORS")):
    """(kind, entity, body) for each send of one of ``kinds``, in send order."""
    out = []
    for entity, msg in c.comm.sent:
        kind = getattr(msg.message_type, "name", msg.message_type)
        if kind in kinds:
            out.append((kind, entity, msg.body))
    return out


PROMPT, TOKEN = 101, 202


def plan_step(model, plan, k, signals) -> ForwardPassArgs:
    """What the conductor sends a request entering ``plan[k]``: its edges and
    step metadata, by the conductor's own `_plan_step`."""
    c = Conductor.__new__(Conductor)
    c._walk_inputs, _, _ = _walk_io(model)
    return c._plan_step(SimpleNamespace(persist_signals=signals), PartitionState("p", plan=list(plan)), k)


def released_by_step(model, plan, k, signals) -> list[TensorPointerInfo]:
    """What the conductor releases as a request's step ``plan[k]`` goes out, by
    its own `_consumed`."""
    c = Conductor.__new__(Conductor)
    c._walk_inputs, c._walk_persists, _ = _walk_io(model)
    state = PartitionState("p", plan=list(plan), cursor=k)
    return c._consumed(SimpleNamespace(persist_signals=signals, partition_states={"p": state}), ["p"])


def _decoder_walks(prefills=("prefill",)):
    """Prefill walks that persist the sampled token, and a decode loop."""
    walks = {walk: GraphNode(name="llm", input_names=["text_inputs", *extra], outputs=[
        GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", persist=True),
    ]) for walk, extra in prefills}
    walks["decode"] = Loop(name="decode_loop", max_iters=8, outputs=[], section=GraphNode(
        name="llm", input_names=["text_inputs"], outputs=[GraphEdge(next_node="llm", name="text_inputs")],
    ))
    return walks


DECODE = WalkStep("decode", bind={"text_inputs": "new_token"})
# the prefill that samples the reply's first token
LAST_PREFILL = {FINAL_PREFILL: True}
PROMPT_KV = {"kv_layout": {"kv": {"main": LabelLayout(
    spans=(Span("ids", "prefill", 4, 4, "text_inputs"),), growth=(("decode_loop", 1),),
).row()}}}
REF, TAIL, LANG, EMB = 303, 404, 505, 606


def _p2():
    model = _ToyModel(_decoder_walks([("prefill", ())]),
                      {"default": [WalkStep("prefill", step_metadata=LAST_PREFILL), DECODE]},
                      {"default": ["prefill", "decode"]}, growth=True)
    return model, {"text_inputs": [PROMPT]}, PROMPT_KV, [
        ("default", "prefill", {"new_token": TOKEN}, False), ("default", "decode", {}, False),
    ], [
        ("NEW_REQUEST", "default", "prefill", {("llm", "text_inputs"): [PROMPT]}, True),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        ("INPUT_SIGNALS", "default", "decode", {("llm", "text_inputs"): [TOKEN]}, False),
        ("UNPERSIST_TENSORS", "worker_0", {TOKEN: 1}),
    ]


def _chunked_prefill():
    model, signals, kwargs, script, sends = _p2()
    model._chunked = ("prefill",)
    tick = chunk_tick_name("llm", "prefill")
    sends[0] = ("NEW_REQUEST", "default", "prefill", {("llm", "text_inputs"): [PROMPT], ("llm", tick): []}, True)
    return model, signals, kwargs, script, sends


def _rebound_name():
    """Whisper's detect -> prompt -> decode: each prefill persists ``new_token``,
    and the prompt walk's own re-persist means it consumes the language token."""
    model = _ToyModel(
        _decoder_walks([("detect", ()), ("prompt", ("tail",))]),
        {"default": [
            WalkStep("detect"), WalkStep("prompt", bind={"text_inputs": "new_token"}, step_metadata=LAST_PREFILL),
            DECODE,
        ]},
        {"default": ["detect", "prompt", "decode"]}, growth=True,
    )
    return model, {"text_inputs": [PROMPT], "tail": [_info(TAIL)]}, PROMPT_KV, [
        ("default", "detect", {"new_token": LANG}, False), ("default", "prompt", {"new_token": TOKEN}, False),
        ("default", "decode", {}, False),
    ], [
        # the tail is read later; only the step before decode samples a real token
        ("NEW_REQUEST", "default", "detect", {("llm", "text_inputs"): [PROMPT]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        ("INPUT_SIGNALS", "default", "prompt", {("llm", "text_inputs"): [LANG], ("llm", "tail"): [TAIL]}, True),
        ("UNPERSIST_TENSORS", PREPROCESS, {TAIL: 1}),
        ("UNPERSIST_TENSORS", "worker_0", {LANG: 1}),
        ("INPUT_SIGNALS", "default", "decode", {("llm", "text_inputs"): [TOKEN]}, False),
        ("UNPERSIST_TENSORS", "worker_0", {TOKEN: 1}),
    ]


def _diffusion(refine=False):
    """An encode walk, then a denoise loop whose loop-back seed arrives empty.
    With LTX-2.5's refine, a second denoise walk reads the conditioning again,
    so the first one's send keeps it."""
    def denoise():
        return Sequential([
            Loop(name="denoise", max_iters=4, outputs=[GraphEdge(next_node="vae", name="latents")],
                 section=GraphNode(name="dit", input_names=["emb", "latents"],
                                   outputs=[GraphEdge(next_node="dit", name="latents")])),
            GraphNode(name="vae", input_names=["latents"], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="image", output_modality="image"),
            ]),
        ])

    walks = {
        "encode": GraphNode(name="enc", input_names=["text_inputs"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="emb", persist=True),
        ]),
        "generate": denoise(),
        "refine": denoise(),
    }
    gens = ["generate", "refine"] if refine else ["generate"]
    md = {"steps": 4}
    model = _ToyModel(walks, {"default": [WalkStep(w, step_metadata=md) for w in ["encode", *gens]]},
                       {"default": list(walks)})
    seeds = {("dit", "emb"): [EMB], ("dit", "latents"): []}
    return model, {"text_inputs": [PROMPT]}, {}, [
        ("default", "encode", {"emb": EMB}, False), *(("default", w, {}, False) for w in gens),
    ], [
        ("NEW_REQUEST", "default", "encode", {("enc", "text_inputs"): [PROMPT]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        *(("INPUT_SIGNALS", "default", w, seeds, False) for w in gens),
        ("UNPERSIST_TENSORS", "worker_0", {EMB: len(gens)}),
    ]


def _fed_loop():
    """V-JEPA 2's rollout: the encoder ahead of the loop feeds its first
    iteration the name the loop then carries, so that loop-back gets no seed."""
    walks = {"rollout": Sequential([
        GraphNode(name="enc", input_names=["video"], outputs=[GraphEdge(next_node="roll", name="hidden")]),
        Loop(name="rollout_loop", max_iters=4, outputs=[], section=GraphNode(
            name="roll", input_names=["hidden"], outputs=[
                GraphEdge(next_node="roll", name="hidden"),
                GraphEdge(next_node=EMIT_TO_CLIENT, name="pred", output_modality="video"),
            ],
        )),
    ])}
    model = _ToyModel(walks, {"default": [WalkStep("rollout")]}, {"default": ["rollout"]})
    return model, {"video": [PROMPT]}, {}, [("default", "rollout", {}, False)], [
        ("NEW_REQUEST", "default", "rollout", {("enc", "video"): [PROMPT]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
    ]


def _stream_consumer(consumer_active=True):
    """Chatterbox's S3Gen: a stream consumer that also reads the reference clip
    its producer reads. The clip goes to both on entry and is released once
    both have read it; each later chunk's pass carries a signal-only edge."""
    walks = {
        "a": GraphNode(name="na", input_names=["text_inputs", "ref"], outputs=[
            StreamingGraphEdge(next_node="nb", name="tok", target_partition="B"),
        ]),
        "b": GraphNode(name="nb", input_names=["tok", "ref"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio"),
        ]),
    }
    model = _ToyModel(walks, {"A": [WalkStep("a")], "B": [WalkStep("b")] if consumer_active else []},
                       {"A": ["a"], "B": ["b"]},
                       connections=[Connection("A", "B", "tok", lambda: FixedChunkPolicy(chunk_size=1))])
    if not consumer_active:
        return model, {"text_inputs": [PROMPT], "ref": [_info(REF)]}, {}, [("A", "a", {}, True)], [
            ("NEW_REQUEST", "A", "a", {("na", "ref"): [REF], ("na", "text_inputs"): [PROMPT]}, False),
            ("NEW_REQUEST", "B", "", {}, False),
            ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1, REF: 1}),
        ]
    return model, {"text_inputs": [PROMPT], "ref": [_info(REF)]}, {}, [
        ("B", "b", {}, False), ("A", "a", {}, True), ("B", "b", {}, True),
    ], [
        ("NEW_REQUEST", "A", "a", {("na", "ref"): [REF], ("na", "text_inputs"): [PROMPT]}, False),
        ("NEW_REQUEST", "B", "b", {("nb", "ref"): [REF]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1, REF: 2}),
        ("INPUT_SIGNALS", "B", "b", {("nb", "ref"): []}, False),
        ("INPUT_SIGNALS", "B", "b", {}, False),  # producer_done
    ]


def _held_for_a_later_step():
    """The reference clip a chunk step reads is still held for the producer's
    later step: each later chunk's pass carries a signal-only edge, so the
    clip is counted once per step that read it."""
    walks = {
        "a0": GraphNode(name="na", input_names=["text_inputs"], outputs=[
            StreamingGraphEdge(next_node="nb", name="tok", target_partition="B"),
        ]),
        "a1": GraphNode(name="na", input_names=["ref"], outputs=[]),
        "b": GraphNode(name="nb", input_names=["tok", "ref"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio"),
        ]),
    }
    model = _ToyModel(walks, {"A": [WalkStep("a0"), WalkStep("a1")], "B": [WalkStep("b")]},
                      {"A": ["a0", "a1"], "B": ["b"]},
                      connections=[Connection("A", "B", "tok", lambda: FixedChunkPolicy(chunk_size=1))])
    return model, {"text_inputs": [PROMPT], "ref": [_info(REF)]}, {}, [
        ("B", "b", {}, False), ("B", "b", {}, False), ("A", "a0", {}, False), ("A", "a1", {}, False),
        ("B", "b", {}, True),
    ], [
        ("NEW_REQUEST", "A", "a0", {("na", "text_inputs"): [PROMPT]}, False),
        ("NEW_REQUEST", "B", "b", {("nb", "ref"): [REF]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        ("INPUT_SIGNALS", "B", "b", {("nb", "ref"): []}, False),
        ("INPUT_SIGNALS", "B", "b", {("nb", "ref"): []}, False),
        ("INPUT_SIGNALS", "A", "a1", {("na", "ref"): [REF]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {REF: 2}),
        ("INPUT_SIGNALS", "B", "b", {}, False),  # producer_done
    ]


def _stream_moved_walks():
    """Qwen3-Omni's Talker: the stream moves it through its prefill walks; the
    conductor sends only the decode seed, after the last prefill reports."""
    walks = {
        "think": GraphNode(name="na", input_names=["text_inputs"], outputs=[
            StreamingGraphEdge(next_node="nb", name="states", target_partition="T"),
        ]),
        "t_prefill": GraphNode(name="nb", input_names=["states"], outputs=[]),
        "t_last": GraphNode(name="nb", input_names=["states"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="embeds", persist=True),
        ]),
        "t_decode": Loop(name="talk_loop", max_iters=8, outputs=[], section=GraphNode(
            name="nb", input_names=["embeds"], outputs=[GraphEdge(next_node="nb", name="embeds")],
        )),
    }
    model = _ToyModel(
        walks, {"A": [WalkStep("think")], "T": [WalkStep(w) for w in ("t_prefill", "t_last", "t_decode")]},
        {"A": ["think"], "T": ["t_prefill", "t_last", "t_decode"]},
        connections=[Connection("A", "T", "states", lambda: FixedChunkPolicy(chunk_size=1),
                                consumer_walk=lambda ctx: "t_prefill")],
    )
    return model, {"text_inputs": [PROMPT]}, {}, [
        ("T", "t_prefill", {}, False), ("T", "t_prefill", {}, False), ("A", "think", {}, True),
        ("T", "t_last", {"embeds": EMB}, False), ("T", "t_decode", {}, False),
    ], [
        ("NEW_REQUEST", "A", "think", {("na", "text_inputs"): [PROMPT]}, False),
        ("NEW_REQUEST", "T", "t_prefill", {}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1}),
        ("INPUT_SIGNALS", "T", "t_prefill", {}, False),  # producer_done, the walk still the stream's
        ("INPUT_SIGNALS", "T", "t_decode", {("nb", "embeds"): [EMB]}, False),
        ("UNPERSIST_TENSORS", "worker_1", {EMB: 1}),
    ]


def _every_partitions_reads():
    """Three partitions, each on a rank of its own: a name two of them read is
    released at ingest once, counted per rank; a partition the request doesn't
    use reads nothing; a signal no step reads is never sent, so it has no
    reader count and stays held."""
    shared, only_a, also_inactive, unread = 1, 2, 3, 4
    walks = {
        "a": GraphNode(name="na", input_names=["x", "y", "z"], outputs=[]),
        "b": GraphNode(name="nb", input_names=["x"], outputs=[]),
        "c": GraphNode(name="nc", input_names=["z"], outputs=[]),
    }
    model = _ToyModel(walks, {"A": [WalkStep("a")], "B": [WalkStep("b")], "C": []},
                      {"A": ["a"], "B": ["b"], "C": ["c"]})
    signals = {"x": [shared], "y": [only_a], "z": [also_inactive], "w": [unread]}
    return model, signals, {}, [("A", "a", {}, False), ("B", "b", {}, False)], [
        ("NEW_REQUEST", "A", "a", {("na", "x"): [shared], ("na", "y"): [only_a], ("na", "z"): [also_inactive]}, False),
        ("NEW_REQUEST", "B", "b", {("nb", "x"): [shared]}, False),
        ("NEW_REQUEST", "C", "", {}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {shared: 2, only_a: 1, also_inactive: 1}),
        # the request's end lets go of what it still holds: no reader count, so nothing to count down
        ("UNPERSIST_TENSORS", PREPROCESS, {}),
    ]


def _another_partitions_persist():
    """A's walk persists ``x`` afresh and B's second step reads it. B may enter
    that step before A's walk writes ``x``, so what is held now is kept for B."""
    walks = {
        "a": GraphNode(name="na", input_names=["x"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="x", persist=True),
        ]),
        "b0": GraphNode(name="nb", input_names=["y"], outputs=[]),
        "b1": GraphNode(name="nb", input_names=["x"], outputs=[]),
    }
    model = _ToyModel(walks, {"A": [WalkStep("a")], "B": [WalkStep("b0"), WalkStep("b1")]},
                      {"A": ["a"], "B": ["b0", "b1"]})
    return model, {"x": [REF], "y": [TAIL]}, {}, [
        ("B", "b0", {}, False), ("A", "a", {"x": EMB}, False), ("B", "b1", {}, False),
    ], [
        ("NEW_REQUEST", "A", "a", {("na", "x"): [REF]}, False),
        ("NEW_REQUEST", "B", "b0", {("nb", "y"): [TAIL]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {TAIL: 1}),
        ("INPUT_SIGNALS", "B", "b1", {("nb", "x"): [REF]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {REF: 2}),
        # nothing reads what A persisted; the request's end lets go of it
        ("UNPERSIST_TENSORS", "worker_0", {}),
    ]


def _persisted_again_while_held():
    """Whisper's ``ts_rules`` with the language detected: ``x`` is held from
    the start, a step that doesn't read it runs, then one whose walk persists
    ``x`` afresh, and the last reads it. That one gets the new value alone."""
    walks = {
        "w0": GraphNode(name="n", input_names=["y"], outputs=[]),
        "w1": GraphNode(name="n", input_names=["y"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="x", persist=True),
        ]),
        "w2": GraphNode(name="n", input_names=["x"], outputs=[]),
    }
    model = _ToyModel(walks, {"default": [WalkStep(w) for w in walks]}, {"default": list(walks)})
    return model, {"x": [REF], "y": [TAIL]}, {}, [
        ("default", "w0", {}, False), ("default", "w1", {"x": EMB}, False), ("default", "w2", {}, False),
    ], [
        ("NEW_REQUEST", "default", "w0", {("n", "y"): [TAIL]}, False),
        ("INPUT_SIGNALS", "default", "w1", {("n", "y"): [TAIL]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {TAIL: 2}),
        # the initial x was never sent, so it has no reader count to release
        ("INPUT_SIGNALS", "default", "w2", {("n", "x"): [EMB]}, False),
        ("UNPERSIST_TENSORS", "worker_0", {EMB: 1}),
    ]


def _exit_jump(exit_name):
    """``_persisted_again_while_held`` with an exit on its first step that skips the walk
    persisting ``x`` afresh. Down that path nothing replaces the held ``x``, so it
    is kept for the last step even though the default path shields it."""
    model, signals, kwargs, _, _ = _persisted_again_while_held()
    plan = [WalkStep("w0", exits={"skip": 2}), WalkStep("w1"), WalkStep("w2")]
    model = _ToyModel(model.get_graph_walk_graphs(), {"default": plan}, {"default": ["w0", "w1", "w2"]})
    jumped = exit_name == "skip"
    script = [("default", "w0", {}, False, exit_name)] + [
        ("default", "w2", {}, False) if jumped else ("default", "w1", {"x": EMB}, False),
    ] + ([] if jumped else [("default", "w2", {}, False)])
    if jumped:
        expected = [
            ("NEW_REQUEST", "default", "w0", {("n", "y"): [TAIL]}, False),
            ("INPUT_SIGNALS", "default", "w2", {("n", "x"): [REF]}, False),
            ("UNPERSIST_TENSORS", PREPROCESS, {REF: 1}),
            # the skipped step still reads ``y`` on the default path, so it is held to the end
            ("UNPERSIST_TENSORS", PREPROCESS, {TAIL: 1}),
        ]
    else:  # an exit the step does not list: the next step, as without one
        expected = _persisted_again_while_held()[4]
    return model, signals, kwargs, script, expected


def _streamed_middle():
    """Qwen3.5 on speech: the history's prefill, the transcript's chunks (a walk the stream feeds), then the
    suffix's prefill, which the stream's end moves on to; the partition ends with the last step."""
    walks = {
        "a": GraphNode(name="na", input_names=["text_inputs"], outputs=[
            StreamingGraphEdge(next_node="nb", name="tok", target_partition="B")]),
        "h": GraphNode(name="nb", input_names=["hist"], outputs=[]),
        # `go` is the empty edge the conductor sends with each pass, which tells the worker its walk
        "c": GraphNode(name="nb", input_names=["tok", "go"], outputs=[]),
        "s": GraphNode(name="nb", input_names=["suf"], outputs=[]),
    }
    model = _ToyModel(walks, {"A": [WalkStep("a")], "B": [WalkStep("h"), WalkStep("c"), WalkStep("s")]},
                      {"A": ["a"], "B": ["h", "c", "s"]},
                      connections=[Connection("A", "B", "tok", lambda: FixedChunkPolicy(chunk_size=1))])
    return model, {"text_inputs": [PROMPT], "hist": [REF], "suf": [TAIL]}, {}, [
        ("B", "h", {}, False), ("B", "c", {}, False), ("A", "a", {}, True), ("B", "c", {}, True), ("B", "s", {}, False),
    ], [
        ("NEW_REQUEST", "A", "a", {("na", "text_inputs"): [PROMPT]}, False),
        ("NEW_REQUEST", "B", "h", {("nb", "hist"): [REF]}, False),
        ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1, REF: 1}),
        ("INPUT_SIGNALS", "B", "c", {("nb", "go"): []}, False),   # the history is in; the stream's step
        ("INPUT_SIGNALS", "B", "c", {("nb", "go"): []}, False),   # a chunk's pass
        ("INPUT_SIGNALS", "B", "c", {}, False),                    # producer_done
        ("INPUT_SIGNALS", "B", "s", {("nb", "suf"): [TAIL]}, False),   # the stream's end moved the plan on
        ("UNPERSIST_TENSORS", PREPROCESS, {TAIL: 1}),
    ]


def _tool_round(rounds, ending="answer"):
    """Qwen3.5 with tools: prefill, decode, a router walk, a tool walk, then a prefill that continues the
    reply and comes back to decode; the router ends the plan with ``answer`` when the turn made no call, or
    with ``speak`` when the turn is spoken.
    Each walk persists what the next reads and replaces its predecessor's, so a round releases the last
    one's tensors."""
    walks = {
        "p": GraphNode(name="n", input_names=["x"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="first", persist=True)]),
        "d": GraphNode(name="n", input_names=["first"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="turn", persist=True)]),
        "r": GraphNode(name="router", input_names=["first", "turn"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="calls", persist=True)]),
        "t": GraphNode(name="tool", input_names=["calls"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="tool_text", persist=True)]),
        "pc": GraphNode(name="n", input_names=["x"], outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name="first", persist=True)]),
    }
    plan = [WalkStep("p"), WalkStep("d"), WalkStep("r", exits={"answer": 5, "speak": 5, "tools": 3}), WalkStep("t"),
            WalkStep("pc", bind={"x": "tool_text"}, exits={"": 1})]
    model = _ToyModel(walks, {"default": plan}, {"default": list(walks)})
    script = [("default", "p", {"first": 601}, False)]
    expected = [("NEW_REQUEST", "default", "p", {("n", "x"): [PROMPT]}, False),
                ("UNPERSIST_TENSORS", PREPROCESS, {PROMPT: 1})]
    uuid = 602
    for _ in range(rounds + 1):
        # a decode, then the router: it sends the turn's calls to the tool, or ends the plan
        last = _ == rounds
        script += [("default", "d", {"turn": uuid}, False),
                   ("default", "r", {"calls": uuid + 1}, False, ending if last else "tools")]
        first = 601 if _ == 0 else uuid - 1
        expected += [("INPUT_SIGNALS", "default", "d", {("n", "first"): [first]}, False),
                     ("INPUT_SIGNALS", "default", "r", {("router", "first"): [first], ("router", "turn"): [uuid]},
                      False),
                     ("UNPERSIST_TENSORS", "worker_0", {first: 2, uuid: 1})]
        if last:
            # the answer's calls are read by no one; the request's end lets go of them
            expected.append(("UNPERSIST_TENSORS", "worker_0", {}))
            break
        script += [("default", "t", {"tool_text": uuid + 2}, False), ("default", "pc", {"first": uuid + 3}, False)]
        expected += [("INPUT_SIGNALS", "default", "t", {("tool", "calls"): [uuid + 1]}, False),
                     ("UNPERSIST_TENSORS", "worker_0", {uuid + 1: 1}),
                     ("INPUT_SIGNALS", "default", "pc", {("n", "x"): [uuid + 2]}, False),
                     ("UNPERSIST_TENSORS", "worker_0", {uuid + 2: 1})]
        uuid += 4
    return model, {"x": [PROMPT]}, {}, script, expected


_PLANS = {
    "a stream's end moves the plan on": _streamed_middle,
    "a tool round comes back to decode": lambda: _tool_round(1),
    "an answer with no tool call ends the plan": lambda: _tool_round(0),
    "a spoken turn ends the plan": lambda: _tool_round(0, "speak"),
    "an exit jumps over the step that re-persists a name": lambda: _exit_jump("skip"),
    "an unlisted exit takes the next step": lambda: _exit_jump("typo"),
    "every partition's reads": _every_partitions_reads,
    "another partition's persist keeps it": _another_partitions_persist,
    "prefill then decode": _p2,
    "chunked prefill seeds its tick": _chunked_prefill,
    "a re-persisted name": _rebound_name,
    "a name persisted again while held": _persisted_again_while_held,
    "diffusion's empty seed": _diffusion,
    "a later reader keeps it": lambda: _diffusion(refine=True),
    "a loop fed ahead of it": _fed_loop,
    "stream consumer re-sends": _stream_consumer,
    "inactive consumer": lambda: _stream_consumer(consumer_active=False),
    "a chunk step's input held for a later step": _held_for_a_later_step,
    "walks the stream moves": _stream_moved_walks,
}


def _play(model, signals, kwargs, script, tmp_path, monkeypatch, counts=None):
    """A conductor given the request, then each report of ``script``; ``counts`` is the tokens each walk's
    report counts."""
    walks = model.get_graph_walk_graphs()
    # each partition on a rank of its own
    groups = [sorted({n for w in p.graph_walks for n in walks[w].get_nodes()}) for p in model.get_partitions()]
    c = _conductor(model, [{"node_names": g, "ranks": [i]} for i, g in enumerate(groups)], tmp_path, monkeypatch)
    worker = {n: f"worker_{i}" for i, g in enumerate(groups) for n in g}
    signals = {name: [i if isinstance(i, TensorPointerInfo) else _info(i) for i in infos]
               for name, infos in signals.items()}
    reports = []
    for partition, walk, persist, done, *exit_name in script:
        node = next(iter(walks[walk].get_nodes()))
        wg = c.node_walk_to_wg[(node, walk)].worker_graph_id
        reports.append([_msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
            request_id="r", worker_graph_ids=[wg], is_first_tp_rank=True, partition_name=partition,
            partition_done=done, graph_walk=walk, exit=exit_name[0] if exit_name else "",
            new_token_counts={"new_token": (counts or {}).get(walk, 0)},
            persist_signals={name: [_info(u, entity=worker[node], node=node, walk=walk)]
                             for name, u in persist.items()},
        ))])
    _run(c, [[_msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
        request_id="r", initial_signals=signals, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs=kwargs,
    ))], *reports])
    return c


@pytest.mark.parametrize("case", list(_PLANS))
def test_a_walk_plan_sends_each_steps_inputs_and_releases_their_last_read(tmp_path, monkeypatch, case):
    model, signals, kwargs, script, expected = _PLANS[case]()
    c = _play(model, signals, kwargs, script, tmp_path, monkeypatch)

    got = []
    for kind, entity, body in _sends(c, ("NEW_REQUEST", "INPUT_SIGNALS", "UNPERSIST_TENSORS")):
        if kind == "UNPERSIST_TENSORS":
            got.append((kind, entity, dict(body.uuid_to_ref_count)))
            continue
        info = body.request_info
        edges = body.initial_inputs if kind == "NEW_REQUEST" else body.inputs
        got.append((kind, info.partition_name, info.graph_walk,
                    {(e.next_node, e.name): [i.uuid for i in e.tensor_info] for e in edges},
                    bool(info.step_metadata.get(FINAL_PREFILL))))
    assert got == expected
    assert any(getattr(msg, "message_type", None) == "request_complete" for _, msg in c.comm.sent), \
        "the plan's last step did not end the request"


def test_a_stop_checks_end_at_a_step_the_stream_feeds_ends_the_partition(tmp_path, monkeypatch):
    """The same report as a stream's end, but with the stop check's END_PARTITION exit: the first token of
    the reply was a stop token, so the suffix's step is not run."""
    from mstar.model.submodule_base import END_PARTITION

    model, signals, kwargs, script, _ = _streamed_middle()
    script = [*script[:3], ("B", "c", {}, True, END_PARTITION)]
    c = _play(model, signals, kwargs, script, tmp_path, monkeypatch)
    assert [(b.request_info.graph_walk) for _, _, b in _sends(c, ("INPUT_SIGNALS",))] == ["c", "c", "c"]
    assert any(getattr(msg, "message_type", None) == "request_complete" for _, msg in c.comm.sent)


def test_a_plan_that_comes_back_runs_under_the_budget_it_has_left(tmp_path, monkeypatch):
    """Each step of a tool round's plan gets what is left of the output budget, so the rounds' decode
    steps together stay in it."""
    model, signals, kwargs, script, _ = _tool_round(1)
    kwargs = {"max_output_tokens": 10}
    c = _play(model, signals, kwargs, script, tmp_path, monkeypatch, counts={"p": 1, "d": 2, "t": 2, "pc": 1})
    sent = [(b.request_info.graph_walk, b.request_info.max_tokens) for _, _, b in _sends(c, ("INPUT_SIGNALS",))]
    assert sent == [("d", 9), ("r", 7), ("t", 7), ("pc", 5), ("d", 4), ("r", 2)]


def _tok_stream():
    return [Connection("A", "B", "tok", lambda: FixedChunkPolicy(chunk_size=1))]


_STREAMED = {"A": ["a"], "B": ["b", "c"]}
_REFUSED = {
    # no worker would ever report on it, so nothing would end it; the request's
    # content can lead there (a model with no walk for its modalities)
    "no partition has a walk": ({"default": []}, None, (), 400, "no walk to run"),
    # a model bug: a walk the model doesn't have fails the same check
    "a walk of another partition": ({"A": [WalkStep("b")], "B": []}, {"A": ["a"], "B": ["b"]}, (), 500,
                                    "not one of its walks"),
    "a bind key the walk doesn't take": ({"default": [WalkStep("a", bind={"y": "x"})]}, None, (), 500,
                                         "not inputs of that walk"),
    # nothing tells B's worker its walk changed
    "a later step only a stream feeds": ({"A": [WalkStep("a")], "B": [WalkStep("b"), WalkStep("c")]}, _STREAMED,
                                         _tok_stream(), 500, "takes no input the conductor sends"),
    # an exit is an edge of the plan, or its end
    "an exit past the plan's end": ({"default": [WalkStep("a", exits={"x": 2})]}, None, (), 500, "or its end"),
    # the stream-moved lookup finds a step by walk name, going forward
    "a back edge where a stream moves the walk": ({"A": [WalkStep("a")], "B": [WalkStep("c", exits={"x": 0})]},
                                                  _STREAMED, [Connection(
                                                      "A", "B", "tok", lambda: FixedChunkPolicy(chunk_size=1),
                                                      consumer_walk=lambda ctx: "c")], 500, "that lead back"),
}


@pytest.mark.parametrize("case", list(_REFUSED))
def test_a_plan_the_conductor_cant_run_is_refused(tmp_path, monkeypatch, caplog, case):
    """Before any worker hears of the request: a 400 when the request leads
    there, and a 500 whose text only the log has when the model's plan is wrong.
    With ``connections``, A's walk streams ``tok`` to B's walk ``c``, a bare
    node run once per chunk."""
    plans, groups, connections, status, message = _REFUSED[case]
    walks = {name: GraphNode(name=f"n{name}", input_names=["x"], outputs=[]) for name in ("a", "b")}
    if connections:
        walks["a"] = GraphNode(name="na", input_names=["x"], outputs=[
            StreamingGraphEdge(next_node="nc", name="tok", target_partition="B"),
        ])
        walks["c"] = GraphNode(name="nc", input_names=["tok"], outputs=[])
    nodes = sorted({n for w in walks.values() for n in w.get_nodes()})
    c = _conductor(_ToyModel(walks, plans, groups, connections=connections),
                   [{"node_names": nodes, "ranks": [0]}], tmp_path, monkeypatch)
    _run(c, [[_new_request("r", {"x": [_info(PROMPT)]})]])

    failed = [m.body for _, m in c.comm.sent if getattr(m, "message_type", None) == "request_failed"]
    assert [(f.request_id, f.status) for f in failed] == [("r", status)]
    assert message in (failed[0].error_message if status < 500 else caplog.text)
    assert not _sends(c, ("NEW_REQUEST",)) and "r" not in c.requests


def test_a_stream_moved_report_of_a_walk_its_plan_is_past_moves_nothing():
    """Only a lagging TP rank's report can name such a walk; with no pass number on the wire it
    can't be told apart, so it neither fails the request nor moves the plan. A walk the plan
    never ran is a bug, and fails it."""
    c = Conductor.__new__(Conductor)
    c.model, c.producer_triggered_partitions = SimpleNamespace(), {"B"}
    c._walk_inputs, c._repeating_walks = {"x": (), "y": (("t", "n"),)}, frozenset()
    # the plan is at its last step; the report names its first
    pstate = PartitionState("B", plan=[WalkStep("x"), WalkStep("y"), WalkStep("z")], cursor=2)
    request = SimpleNamespace(partition_states={"B": pstate}, persist_signals={})
    pstate.graph_walk = "x"
    args = c._next_plan_args(request, "B", done=False)
    assert not args.request_done and not args.inputs and pstate.cursor == 2
    pstate.graph_walk = "w"
    with pytest.raises(RuntimeError, match="which its plan doesn't run"):
        c._next_plan_args(request, "B", done=False)


def test_a_stream_moved_partitions_exit_leads_from_the_step_its_walk_resolves_to():
    """The report names a walk, not a step: the exit is looked up on the step the
    conductor resolves that walk to, and leads past the step after it."""
    c = Conductor.__new__(Conductor)
    c.model, c.producer_triggered_partitions = SimpleNamespace(), {"B"}
    c._walk_inputs = {w: (("t", "n"),) for w in ("x", "y", "z")}
    c._walk_persists, c._repeating_walks = {w: frozenset() for w in "xyz"}, frozenset()
    pstate = PartitionState("B", plan=[WalkStep("x", exits={"skip": 2}), WalkStep("y"), WalkStep("z")], cursor=0)
    request = SimpleNamespace(partition_states={"B": pstate}, persist_signals={})
    pstate.graph_walk, pstate.exit = "x", "skip"
    args = c._next_plan_args(request, "B", done=False)
    assert pstate.cursor == 2 and pstate.graph_walk == "z" and args.inputs


def test_two_ranks_naming_different_exits_in_one_pass_fail_the_request():
    c = Conductor.__new__(Conductor)
    c.enable_prof, c.producer_triggered_partitions = False, set()
    pstate = PartitionState("B")
    c.requests = {"r": SimpleNamespace(
        partition_states={"B": pstate}, output_chunk_counts={}, streaming_connections={})}
    body = dict(request_id="r", worker_graph_ids=[], partition_name="B", is_first_tp_rank=True)
    c._process_worker_graphs_done(WorkerGraphsDone(exit="a", **body))
    assert pstate.exit == "a"
    c._process_worker_graphs_done(WorkerGraphsDone(exit="a", **body))  # the same one from another report
    with pytest.raises(RuntimeError, match="exits 'a' and 'b'"):
        c._process_worker_graphs_done(WorkerGraphsDone(exit="b", **body))


# --- agents: a plan that spawns children, holds a join step for them, and tears them down after it -----------


def _spawning_model():
    """The parent's plan: prefill, decode, router (exit ``spawn``), the tool walk that joins the children (held
    until they end), a continuation prefill, back to decode. An agent's: prefill, decode, router (it answers)."""
    def walk(node, inputs, persists=()):
        return GraphNode(name=node, input_names=inputs, outputs=[
            GraphEdge(next_node=EMPTY_DESTINATION, name=name, persist=True) for name in persists])

    walks = {
        "p": walk("n", ["text_inputs"], ["first"]), "d": walk("n", ["first"], ["turn"]),
        "r": walk("router", ["first", "turn"], ["tool_calls", "agent_0", "agent_1", "agent_answer"]),
        "t": walk("tool", ["tool_calls", "agent_answers"], ["tool_text"]),
        "pc": walk("n", ["text_inputs"], ["first"]),
    }
    router = WalkStep("r", exits={"answer": 5, "spawn": 3}, spawns=("agent_0", "agent_1"))
    parent = [WalkStep("p"), WalkStep("d"), router, WalkStep("t", joins=True),
              WalkStep("pc", bind={"text_inputs": "tool_text"}, exits={"": 1})]
    agent = [WalkStep("p"), WalkStep("d"), WalkStep("r", exits={"answer": 3})]
    return _ToyModel(walks, {"default": parent, "agent": agent}, {"default": list(walks)})


def _spawn_batches(parent="r"):
    """The reports of a parent that spawns two agents, which each answer: up to the join."""
    a0, a1 = f"{parent}-agent-0", f"{parent}-agent-1"
    return a0, a1, [
        [_new_request(parent, {"text_inputs": [_info(101)]})],
        [_done_report(parent, "p", {"first": 601})],
        [_done_report(parent, "d", {"turn": 602})],
        [_done_report(parent, "r", {"tool_calls": 603, "agent_0": 611, "agent_1": 612}, exit_name="spawn")],
        [_done_report(a0, "p", {"first": 621}), _done_report(a1, "p", {"first": 631})],
        [_done_report(a0, "d", {"turn": 622}), _done_report(a1, "d", {"turn": 632})],
        [_done_report(a0, "r", {"agent_answer": 623}, exit_name="answer")],
    ]


def _done_report(rid, walk, persist, exit_name="", done=False):
    return _msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
        request_id=rid, worker_graph_ids=[_WGS[walk]], is_first_tp_rank=True, partition_name="default",
        partition_done=done, graph_walk=walk, exit=exit_name,
        persist_signals={name: [_info(u, entity="worker_0", node="n", walk=walk)] for name, u in persist.items()},
    ))


_WGS: dict[str, int] = {}


def _spawn_conductor(tmp_path, monkeypatch):
    model = _spawning_model()
    walks = model.get_graph_walk_graphs()
    nodes = sorted({n for w in walks.values() for n in w.get_nodes()})
    c = _conductor(model, [{"node_names": nodes, "ranks": [0]}], tmp_path, monkeypatch, max_concurrent_requests=1)
    _WGS.update({walk: c.node_walk_to_wg[(next(iter(g.get_nodes())), walk)].worker_graph_id
                 for walk, g in walks.items()})
    return c


def _sent(c, kind):
    return [body for entity, msg in c.comm.sent for body in [msg.body]
            if getattr(msg.message_type, "name", msg.message_type) == kind]


def _joined(c, parent="r"):
    """The inputs the parent's join step was sent: {signal: uuids}, and its step metadata."""
    (body,) = [b for b in _sent(c, "INPUT_SIGNALS") if b.request_id == parent and b.request_info.graph_walk == "t"]
    return {e.name: [i.uuid for i in e.tensor_info] for e in body.inputs}, body.request_info.step_metadata


def _spawned(tmp_path, monkeypatch, name, *more):
    """A conductor after the parent spawned and its first child answered, then the batches ``more`` makes."""
    (tmp_path / name).mkdir()
    c = _spawn_conductor(tmp_path / name, monkeypatch)
    a0, a1, batches = _spawn_batches()
    _run(c, [*batches, *(batch for make in more for batch in make())])
    return c, a0, a1


def test_a_parent_that_spawns_starts_a_child_per_prompt_and_holds_its_join_until_all_end(tmp_path, monkeypatch):
    c, a0, a1 = _spawned(tmp_path, monkeypatch, "one")

    # a child per prompt, on the prompt's tensor; inside the parent's slot
    starts = {b.request_id: b for b in _sent(c, "NEW_REQUEST")}
    assert list(starts) == ["r", a0, a1]
    assert [[i.uuid for i in e.tensor_info] for e in starts[a0].initial_inputs] == [[611]]
    assert c.requests[a0].parent == "r" and c._counted_requests() == 1 and c.requests["r"].children == [a0, a1]
    assert not c.waiting_queue, "a child queued behind the cap its parent holds a slot under"
    # one child has ended; the join waits for the other. The ended one has given up its resources, as any request
    # does when it ends, but keeps its tensors and its handle: its answer is the parent's to read
    assert not [b for b in _sent(c, "INPUT_SIGNALS") if b.request_info.graph_walk == "t"]
    assert a0 not in c.draining and c.requests[a0].ended
    assert [(b.request_id, b.keep_outputs) for b in _sent(c, "REMOVE_REQUEST")] == [(a0, True)]
    assert not _sent(c, "RELEASE_OUTPUTS")


def test_the_join_reads_every_childs_answer_and_the_children_go_only_after_it_has(tmp_path, monkeypatch):
    def last():
        return [[_done_report("r-agent-1", "r", {"agent_answer": 633}, "answer")]]

    c, a0, a1 = _spawned(tmp_path, monkeypatch, "two", last)
    inputs, metadata = _joined(c)
    assert inputs == {"tool_calls": [603], "agent_answers": [623, 633]}
    assert metadata["agents"] == 2 and metadata["agent_errors"] == {}
    assert not _sent(c, "RELEASE_OUTPUTS"), "the children's tensors went before the join step read their answers"
    assert sorted(b.request_id for b in _sent(c, "REMOVE_REQUEST")) == [a0, a1]

    c, a0, a1 = _spawned(tmp_path, monkeypatch, "three", last, lambda: [[_done_report("r", "t", {"tool_text": 641})]])
    assert sorted(b.request_id for b in _sent(c, "RELEASE_OUTPUTS")) == [a0, a1]
    assert a0 not in c.requests and a1 not in c.requests
    # nothing of a child reached the API server, and each prompt was let go of once, by its child
    assert not [m for e, m in c.comm.sent if e == "api_server" and "agent" in str(getattr(m.body, "request_id", ""))]
    freed = [u for body in _sent(c, "UNPERSIST_TENSORS") for u in body.uuid_to_ref_count if u in (611, 612)]
    assert sorted(freed) == [611, 612]


def test_a_child_that_fails_is_an_error_answer_and_one_that_runs_past_its_deadline_is_another(tmp_path, monkeypatch):
    def fail():
        return [[_msg(ConductorMessageType.FAIL_REQUESTS, FailRequests({"r-agent-1": "boom"}))]]

    c, a0, a1 = _spawned(tmp_path, monkeypatch, "fails", fail)
    inputs, metadata = _joined(c)
    assert inputs == {"tool_calls": [603], "agent_answers": [623]}
    assert metadata["agents"] == 2 and metadata["agent_errors"] == {"1": "Error: the agent failed: boom"}
    assert a1 in c.draining, "a failed child is torn down at once"
    # late word from a child that is gone is ignored
    c, a0, a1 = _spawned(tmp_path, monkeypatch, "late", fail, lambda: [
        [_done_report("r-agent-1", "d", {"turn": 999})],
        [_msg(ConductorMessageType.READS_DONE, ReadsDone(request_id="r-agent-1", entity_id="worker_0"))]])
    assert _joined(c)[1]["agent_errors"] == {"1": "Error: the agent failed: boom"}

    monkeypatch.setattr(conductor_mod, "CHILD_SECONDS", 0.0)
    c, a0, a1 = _spawned(tmp_path, monkeypatch, "slow")
    assert _joined(c)[1]["agent_errors"] == {
        "0": "Error: the agent timed out", "1": "Error: the agent timed out"}


def test_a_parent_that_is_aborted_takes_its_children_with_it_and_tells_the_api_server_nothing_of_them(
    tmp_path, monkeypatch,
):
    abort = [[_msg(ConductorMessageType.ABORT_REQUEST, AbortRequest("r"))]]
    c, a0, a1 = _spawned(tmp_path, monkeypatch, "abort", lambda: abort)
    # the running child drains; the one that had ended (its resources already gone) has its tensors and handle freed
    assert {"r", a1} <= set(c.draining)
    assert [b.request_id for b in _sent(c, "RELEASE_OUTPUTS")] == [a0]
    assert not [m for e, m in c.comm.sent if e == "api_server" and "agent" in str(getattr(m.body, "request_id", ""))]
    assert not [b for b in _sent(c, "INPUT_SIGNALS") if b.request_info.graph_walk == "t"], "the join ran"


def _after(tmp_path, monkeypatch, name, *more):
    """A conductor after the parent spawned two agents (none has answered), then the batches ``more`` makes."""
    (tmp_path / name).mkdir()
    c = _spawn_conductor(tmp_path / name, monkeypatch)
    _, _, batches = _spawn_batches()
    _run(c, [*batches[:4], *(batch for make in more for batch in make())])
    return c


def test_a_late_word_for_a_request_that_was_aborted_changes_nothing(tmp_path, monkeypatch):
    """The abort of a parent in the middle of its tool round, and of one in the middle of its join: the reports
    its workers and its children's still send afterwards (a pass that finished, an agent's answer, a failure, the
    reads they were draining) hit requests the conductor no longer has or is tearing down, and are dropped."""
    def abort_then_late():
        return [
            [_msg(ConductorMessageType.ABORT_REQUEST, AbortRequest("r"))],
            [_done_report("r-agent-0", "d", {"turn": 1}), _done_report("r", "r", {"tool_calls": 2}, "answer"),
             _done_report("r-agent-1", "r", {"agent_answer": 3}, "answer"),
             _msg(ConductorMessageType.FAIL_REQUESTS, FailRequests({"r-agent-0": "late", "r": "late"}))],
            *([_msg(ConductorMessageType.READS_DONE, ReadsDone(request_id=rid, entity_id=entity))]
              for rid in ("r", "r-agent-0", "r-agent-1") for entity in ("worker_0", "api_server_preprocess_worker")),
            [],
        ]

    c = _after(tmp_path, monkeypatch, "late", abort_then_late)
    assert not c.requests and not c.draining, "a request the abort took was left behind by a late word"
    assert not [b for b in _sent(c, "INPUT_SIGNALS") if b.request_info.graph_walk == "t"], "the join ran"
    assert not [m for e, m in c.comm.sent if e == "api_server" and getattr(m, "message_type", "") == "request_complete"]


def test_an_agent_that_answers_after_its_parent_is_gone_is_torn_down_not_waited_for(tmp_path, monkeypatch):
    abort = [[_msg(ConductorMessageType.ABORT_REQUEST, AbortRequest("r"))]]
    c = _after(tmp_path, monkeypatch, "orphan", lambda: abort)
    # the parent's drain ends, then its agent's report lands: nothing waits on the agent
    _run(c, [[_msg(ConductorMessageType.READS_DONE, ReadsDone(request_id="r", entity_id="worker_0"))],
             [_msg(ConductorMessageType.READS_DONE,
                   ReadsDone(request_id="r", entity_id="api_server_preprocess_worker"))],
             [_done_report("r-agent-0", "r", {"agent_answer": 5}, "answer")]])
    assert "r" not in c.requests
