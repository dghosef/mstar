"""Tools a model's reply can call, as two nodes of the model's partition.

A model with ``tools`` adds a ``route`` walk of a ``Router`` node and a ``tool`` walk of a
``Tool`` node (`tool_walks`), and a plan that cycles through them (`tool_plan`)::

    prefill -> decode -> route -(tools)-> tool -> prefill that continues the reply -> decode ...
                           `-(answer)-> end
                           `-(speak)-> end
                           `-(spawn)-> join the agents' answers -> prefill that continues the reply -> decode ...

Decode ends on the turn-ending token, as it always does. The router reads the turn's ids
(the first token the prefill sampled, then the decode loop's accumulated ``turn_ids``),
parses the tool calls out of them, persists them as ``tool_calls`` and names the exit:
``tools`` when there are calls, ``answer`` when there are none (the plan ends). A turn whose
calls are all ``speak`` (when the request asks for audio) takes the ``speak`` exit instead: the
router writes the text onto its ``speak_text`` stream edge, which a `MultiModel` link connects to
a speech model, and the plan ends (the turn is spoken, not answered to the model); on every other
exit it writes nothing there. A turn that mixes ``speak`` with other calls runs the others as tools,
and the speech model hears nothing of it. A turn whose calls are all ``spawn_agent`` (once per request, and not in a
child) takes the ``spawn`` exit: the router persists a prompt per task (``agent_0``, ...: the chat template's ids), the
conductor starts a child request of the same model from each (the plan's ``spawns``) and holds the ``join`` step until
all have ended, when the tool node frames their answers as tool responses. A child's router persists its final turn as
``agent_answer`` when it answers. The tool node runs the calls it is
given, each by the handler of its name (`TOOL_HANDLERS`), all of a turn's at once, and
persists ``tool_text``: the rest of the conversation as the chat template writes it, for
the next prefill. A call to a tool with no handler is an error the model reads.

``tool_text`` is two tensors: the ids to write, which start with the turn's last id (its stop
token), and the position that id belongs at. The step speculated past the stop may have
written it already, so the prefill leaves it out when the position counter says so
(`SampledTokenDecoder`). The text counts toward the request's output budget, so a request's
tokens, tool text included, stay within the KV it reserved at intake.

The ``python`` handler runs each call as ``python -I`` in a process of its own, killed on a
timeout, with an empty environment and scratch directory and the CPU time, memory, file size and (for a
server that is not root) process count limited. That is no sandbox: it has no
network or filesystem isolation, and the code is whatever the model wrote. Serve it
only where the model's callers are trusted.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    AGENT_ANSWER,
    AGENT_ANSWERS,
    AGENT_CHILD,
    APPENDS,
    FINAL_PREFILL,
    PREFILL_END,
    TOOLS,
    TURN_IDS,
    CurrentForwardPassInfo,
    WalkStep,
)
from mstar.engine.engine import ExecutingBatch
from mstar.graph.base import GraphEdge, GraphNode, GraphSection
from mstar.graph.special_destinations import EMPTY_DESTINATION
from mstar.model.keys import SPEAK_TEXT
from mstar.model.submodule_base import (
    ARNodeInputs,
    ModelInputsFromEngine,
    NodeSubmodule,
    StopCheck,
)
from mstar.utils.threads import ThreadPool

logger = logging.getLogger(__name__)

ROUTER_NODE = "Router"
ROUTE_WALK = "route"
TOOL_NODE = "Tool"
TOOL_WALK = "tool"
JOIN_WALK = "join"
TOOL_CALLS = "tool_calls"
TOOL_TEXT = "tool_text"
# the exits of the router: a turn with no tool call answers (the plan ends); one with calls runs them
ANSWER = "answer"
TOOLS_EXIT = "tools"
SPEAK = "speak"
SPAWN = "spawn_agent"
SPAWN_EXIT = "spawn"
# the agents: a prompt per child (persisted by the router, one tensor each), the flag that the request has
# spawned, a child's final turn, and the answers the join step's tool node reads
MAX_AGENTS = 4
AGENT_PROMPTS = tuple(f"agent_{i}" for i in range(MAX_AGENTS))
SPAWNED = "spawned"
# the most tokens one agent's answer may bring back to its parent (the rest is cut)
ANSWER_TOKENS = 512

PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Run Python code in a fresh interpreter and read what it prints.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "the program to run"}},
            "required": ["code"],
        },
    },
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, str]


class QwenToolFormat:
    """Qwen3.5's tool grammar: calls in the reply as
    ``<tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>``,
    and results as ``<tool_response>`` blocks of one user turn
    (``chat_template.jinja``)."""

    _CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
    _PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

    def parse(self, text: str) -> list[ToolCall]:
        return [
            ToolCall(name.strip(), {k.strip(): v for k, v in self._PARAMETER.findall(body)})
            for name, body in self._CALL.findall(text)
        ]

    def render_frame(self, enable_thinking: bool) -> tuple[str, str]:
        """The text before and after the results (each `wrap`ped and tokenized apart): what opens the
        user's turn after the stop token that ended the model's, then what closes it and opens the
        assistant's next."""
        opening = "\n<|im_start|>user"
        think = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        closing = "<|im_end|>\n<|im_start|>assistant\n" + think
        return opening, closing

    @staticmethod
    def wrap(result: str) -> str:
        # the template trims a message's content
        return f"\n<tool_response>\n{result.strip()}\n</tool_response>"


def run_python(
    code: str, timeout: float = 10.0, cpu_seconds: int = 10, memory_bytes: int = 1 << 30, max_chars: int = 4000,
    file_bytes: int = 1 << 24,
) -> str:
    """``code`` run by ``python -I`` in a process group of its own, with an empty environment (none of the
    server's keys or tokens) in an empty scratch directory; its output (stdout and stderr), or an error naming
    the limit it hit."""

    # the limits are set by the child itself, before it runs the code (no fork hook in a threaded process)
    wrapper = (
        "import resource, sys\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({memory_bytes}, {memory_bytes}))\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({file_bytes}, {file_bytes}))\n"
        "resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))\n"
        "exec(compile(sys.argv[1], '<tool>', 'exec'))\n"
    )
    with tempfile.TemporaryDirectory(prefix="tool-") as scratch:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", wrapper, code], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True, env={}, cwd=scratch,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return f"Error: the code ran past {timeout:g} seconds and was stopped."
    if len(out) > max_chars:
        out = out[:max_chars] + "\n[output cut]"
    if proc.returncode and not out:
        return f"Error: the code exited with status {proc.returncode}."
    return out


def _python(arguments: Mapping[str, str]) -> str:
    if "code" not in arguments:
        return "Error: python takes the argument code."
    return run_python(arguments["code"])


SPEAK_SCHEMA = {
    "type": "function",
    "function": {
        "name": SPEAK,
        "description": "Say text aloud to the user. Use it when the user asks to hear the answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "what to say"},
                "voice": {"type": "string", "description": "a voice name; leave out for the default"},
            },
            "required": ["text"],
        },
    },
}

SPAWN_SCHEMA = {
    "type": "function",
    "function": {
        "name": SPAWN,
        "description": (
            "Start an agent that works on a task by itself and returns its answer. Call it once, with several "
            "tasks if they are independent: the agents run at the same time. Use it once per conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "everything the agent needs to know"}},
            "required": ["task"],
        },
    },
}

# The tools a serving config's ``tools`` can name: the schema the chat template shows the model, and for
# those the tool node runs, a handler (arguments to the text the model reads back). ``speak`` and
# ``spawn_agent`` have none: the router sends their calls on.
TOOL_HANDLERS: dict[str, Callable[[Mapping[str, str]], str]] = {"python": _python}
TOOL_SCHEMAS = {"python": PYTHON_TOOL_SCHEMA, SPEAK: SPEAK_SCHEMA, SPAWN: SPAWN_SCHEMA}


def check_tools(names: Sequence[str]) -> tuple[str, ...]:
    if unknown := sorted(set(names) - set(TOOL_SCHEMAS)):
        raise ValueError(f"unknown tools {unknown}; the tools are {sorted(TOOL_SCHEMAS)}")
    return tuple(names)


def tool_walks(spawn: bool = False) -> dict[str, GraphSection]:
    """The ``route`` walk (the turn's ids in, the parsed calls out) and the ``tool`` walk (the calls
    in, ``tool_text`` out, persisted and counted toward the request's output budget). With ``spawn`` the
    router also persists a spawn's prompts and flag and a child's answer, and a ``join`` walk of the tool
    node reads the answers of the agents."""
    outputs = [GraphEdge(next_node=EMPTY_DESTINATION, name=TOOL_CALLS, persist=True)]
    if spawn:
        outputs += [
            GraphEdge(next_node=EMPTY_DESTINATION, name=name, persist=True)
            for name in (*AGENT_PROMPTS, SPAWNED, AGENT_ANSWER)
        ]
    def tool(inputs: list[str]) -> GraphNode:
        return GraphNode(
            name=TOOL_NODE, input_names=inputs,
            outputs=[GraphEdge(
                next_node=EMPTY_DESTINATION, name=TOOL_TEXT, persist=True, conductor_new_token=True)],
        )

    walks = {
        ROUTE_WALK: GraphNode(
            name=ROUTER_NODE, input_names=["new_token", TURN_IDS, PREFILL_END, *([SPAWNED] if spawn else [])],
            outputs=outputs,
        ),
        TOOL_WALK: tool([TOOL_CALLS]),
    }
    if spawn:
        # The join is a walk of its own: the answers are held to the request's end, so a tool step that read
        # them would find, after the join, tensors the children's release has freed.
        walks[JOIN_WALK] = tool([TOOL_CALLS, AGENT_ANSWERS])
    return walks


def tool_plan(
    prefills: list[WalkStep], decode_walk: str, prefill_walk: str, enable_thinking: bool, speak: bool = False,
    spawn: bool = False, child: bool = False,
) -> list[WalkStep]:
    """The plan's own ``prefills`` (one, or a spoken prompt's several), decode, the router, the tool, and
    the prefill that continues the reply and comes back to decode; the router's ``answer`` ends the plan.
    ``spawn``: the request may start agents; the plan then ends with their join (the tool step, held until
    the children have ended) and a second continuation prefill. ``child``: the request is an agent, whose
    router persists its final turn for its parent."""
    n = len(prefills)
    end = n + 4 + (2 if spawn else 0)
    metadata = {SPEAK: speak, SPAWN: spawn, AGENT_CHILD: child, "enable_thinking": enable_thinking}
    plan = [
        *(replace(step, step_metadata={**step.step_metadata, TOOLS: True}) for step in prefills),
        WalkStep(decode_walk, bind={"text_inputs": "new_token"}, step_metadata={TOOLS: True}),
        # ``speak``: whether the request has a speech output the router may send a spoken turn's text to
        WalkStep(
            ROUTE_WALK, exits={ANSWER: end, SPEAK: end, TOOLS_EXIT: n + 2, **({SPAWN_EXIT: n + 4} if spawn else {})},
            step_metadata=metadata, spawns=AGENT_PROMPTS if spawn else (),
        ),
        WalkStep(TOOL_WALK, step_metadata={"enable_thinking": enable_thinking}),
        WalkStep(
            prefill_walk, bind={"text_inputs": TOOL_TEXT}, exits={"": n},
            step_metadata={FINAL_PREFILL: True, TOOLS: True, APPENDS: True},
        ),
    ]
    if spawn:
        plan += [
            WalkStep(JOIN_WALK, step_metadata={"enable_thinking": enable_thinking}, joins=True),
            WalkStep(
                prefill_walk, bind={"text_inputs": TOOL_TEXT}, exits={"": n},
                step_metadata={FINAL_PREFILL: True, TOOLS: True, APPENDS: True},
            ),
        ]
    return plan


class _CpuNode(NodeSubmodule):
    """What the router and the tool node share: host work on a CPU rank, a row at a time per batch."""

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        self._out_device = torch.device(device)
        # the device the node's outputs are made on (`get_device` reads a parameter)
        self._probe = torch.nn.Parameter(torch.zeros(1, device=self._out_device), requires_grad=False)

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[ARNodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict:
        return {"rows": [row.kwargs for row in inputs]}

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows=(), **kwargs) -> NameToTensorList:
        (rid,) = engine_inputs.request_ids
        return self.forward_batched(graph_walk, engine_inputs, rows=rows)[rid]

    def _tensor(self, values: list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype).to(self._out_device)


def _pack(payload: dict) -> list[int]:
    return list(json.dumps(payload).encode())


def _unpack(tensor: torch.Tensor) -> dict:
    return json.loads(bytes(tensor.reshape(-1).tolist()))


class ToolRouterSubmodule(_CpuNode):
    """The ``Router`` node: parse the turn's tool calls and say where the plan goes."""

    def __init__(
        self, tokenizer, device: str | torch.device = "cpu", fmt: QwenToolFormat | None = None,
        render_agent: Callable[[str, bool], list[int]] | None = None,
    ):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.format = fmt or QwenToolFormat()
        # a task (and whether it thinks) to the prompt ids of an agent that does it (the chat template, the
        # parent's tools)
        self.render_agent = render_agent

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        first = inputs["new_token"][0].reshape(-1)
        ids = torch.cat([first, *(t.reshape(-1) for t in inputs.get(TURN_IDS, []))]).cpu()
        # the last id of the turn (its stop token) is fed by the decode step that follows it, and
        # the turn's others by the steps before: it belongs at the prefill's end plus all the rest
        start = int(inputs[PREFILL_END][0]) + ids.numel() - 1
        return ARNodeInputs(input_seq_len=0, kwargs={
            "ids": ids, "start": start, SPEAK: bool(fwd_info.step_metadata.get(SPEAK, False)),
            # a spawn is open to a request that has not spawned yet and is not itself an agent
            SPAWN: bool(fwd_info.step_metadata.get(SPAWN, False)) and not inputs.get(SPAWNED),
            "child": bool(fwd_info.step_metadata.get(AGENT_CHILD, False)),
            "thinking": bool(fwd_info.step_metadata.get("enable_thinking", True)),
        })

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows=(), **kwargs,
    ) -> dict[str, NameToTensorList]:
        out = {}
        for rid, row in zip(engine_inputs.request_ids, rows, strict=True):
            ids = row["ids"]
            calls = self.format.parse(self.tokenizer.decode(ids.tolist(), skip_special_tokens=False))
            payload = {
                "tail": int(ids[-1]), "start": row["start"],
                "calls": [[c.name, dict(c.arguments)] for c in calls],
            }
            out[rid] = {TOOL_CALLS: [self._tensor(_pack(payload), torch.uint8)]}
            said = [c.arguments for c in calls if c.name == SPEAK and str(c.arguments.get("text", "")).strip()]
            if row[SPEAK] and calls and len(said) == len(calls):
                # a spoken turn: its text goes to the speech model, one item. Any other turn writes
                # nothing there, so that model hears nothing of it.
                text = " ".join(str(args["text"]).strip() for args in said)
                out[rid][SPEAK_TEXT] = [self._tensor(_pack({"text": text, "voice": said[0].get("voice")}), torch.uint8)]
            tasks = [str(c.arguments.get("task", "")).strip() for c in calls if c.name == SPAWN]
            if row[SPAWN] and self.render_agent is not None and calls and len(tasks) == len(calls) and all(tasks):
                # a spawn: a prompt per task, and the flag that the request has spawned
                for name, task in zip(AGENT_PROMPTS, tasks, strict=False):
                    out[rid][name] = [self._tensor(self.render_agent(task, row["thinking"]), torch.long)]
                out[rid][SPAWNED] = [self._tensor([1], torch.long)]
            elif row["child"] and not calls:
                # an agent's final turn, for its parent
                out[rid][AGENT_ANSWER] = [self._tensor(ids.tolist(), torch.long)]
        return out

    def check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo, outputs: Mapping[str, list[torch.Tensor]],
    ) -> StopCheck:
        if SPEAK_TEXT in outputs:
            return StopCheck(exit=SPEAK)
        if AGENT_PROMPTS[0] in outputs:
            return StopCheck(exit=SPAWN_EXIT)
        return StopCheck(exit=TOOLS_EXIT if _unpack(outputs[TOOL_CALLS][0])["calls"] else ANSWER)


class ToolSubmodule(_CpuNode):
    """The ``Tool`` node: run the calls it is given, each by its tool's handler, and write the next
    prefill's text."""

    MAX_WORKERS = 8

    def __init__(
        self, tokenizer, handlers: Mapping[str, Callable[[Mapping[str, str]], str]],
        device: str | torch.device = "cpu", fmt: QwenToolFormat | None = None,
    ):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.handlers = dict(handlers)
        self.format = fmt or QwenToolFormat()
        self._pool = ThreadPool(max_workers=self.MAX_WORKERS, thread_name_prefix="tool")

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        return ARNodeInputs(input_seq_len=0, kwargs={
            "payload": _unpack(inputs[TOOL_CALLS][0]),
            "enable_thinking": bool(fwd_info.step_metadata.get("enable_thinking", True)),
            "budget": int(fwd_info.max_tokens),
            "answers": self._answers(fwd_info.step_metadata, inputs.get(AGENT_ANSWERS, ())),
        })

    def _answers(self, metadata: Mapping, tensors: Sequence[torch.Tensor]) -> list[str] | None:
        """The agents' answers, in task order, when this step joins them (the conductor says how many agents
        there were, and which of them failed, in the step's metadata); else None."""
        n = int(metadata.get("agents", 0))
        if not n:
            return None
        errors = metadata.get("agent_errors") or {}
        answers = iter(
            self.tokenizer.decode(t.reshape(-1).tolist()[:ANSWER_TOKENS], skip_special_tokens=True).strip()
            for t in tensors
        )
        return [errors[str(i)] if str(i) in errors else next(answers, "Error: the agent gave no answer")
                for i in range(n)]

    def _text_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _run(self, name: str, arguments: Mapping[str, str]) -> str:
        handler = self.handlers.get(name)
        if handler is None:
            return f"Error: there is no tool {name!r}; the tools are {sorted(self.handlers)}."
        try:
            return handler(arguments)
        except Exception as error:  # the model reads it; the request goes on
            return f"Error: {name} failed: {type(error).__name__}: {error}"

    def _text(self, tail: int, results: list[str], enable_thinking: bool, budget: int) -> list[int]:
        """The ids to write: the turn's last id, then the template's text around the results."""
        opening, closing = self.format.render_frame(enable_thinking)
        head, closing_ids = self._text_ids(opening), self._text_ids(closing)
        # what is left of the request's budget bounds the text; the results give way, evenly
        room = budget - 1 - len(head) - len(closing_ids)
        if room < 0 or not results:
            return [tail]
        share = room // len(results)
        body = [t for r in results for t in self._text_ids(self.format.wrap(r))[:share]]
        return [tail, *head, *body, *closing_ids]

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows=(), **kwargs,
    ) -> dict[str, NameToTensorList]:
        # every call of every row at once; a row's results come back together. A row that joins agents
        # has their answers in place of the calls' results
        futures = [[] if row["answers"] is not None else
                   [self._pool.submit(self._run, name, args) for name, args in row["payload"]["calls"]]
                   for row in rows]
        return {
            rid: {TOOL_TEXT: [
                self._tensor(self._text(
                    row["payload"]["tail"],
                    row["answers"] if row["answers"] is not None else [f.result() for f in row_futures],
                    row["enable_thinking"], row["budget"]), torch.long),
                self._tensor([row["payload"]["start"]], torch.long),
            ]}
            for rid, row, row_futures in zip(engine_inputs.request_ids, rows, futures, strict=True)
        }
