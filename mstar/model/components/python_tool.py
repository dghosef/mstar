"""Tools a model's reply can call, as two nodes of the model's partition.

A model with ``tools`` adds a ``route`` walk of a ``Router`` node and a ``tool`` walk of a
``Tool`` node (`tool_walks`), and a plan that cycles through them (`tool_plan`)::

    prefill -> decode -> route -(tools)-> tool -> prefill that continues the reply -> decode ...
                           `-(answer)-> end

Decode ends on the turn-ending token, as it always does. The router reads the turn's ids
(the first token the prefill sampled, then the decode loop's accumulated ``turn_ids``),
parses the tool calls out of them, persists them as ``tool_calls`` and names the exit:
``tools`` when there are calls, ``answer`` when there are none (the plan ends). It is the
only node that decides; a model that routes by something else (a call that speaks, which
a TTS partition takes) adds an exit and its steps. The tool node runs the calls it is
given, each by the handler of its name (`TOOL_HANDLERS`), all of a turn's at once, and
persists ``tool_text``: the rest of the conversation as the chat template writes it, for
the next prefill. A call to a tool with no handler is an error the model reads.

``tool_text`` is ``[turn_len, <stop token>, ...]``: the length of the turn that
produced it, which the prefill measures its request's state against
(`SampledTokenDecoder`), then the ids to write. It counts toward the request's
output budget, so a request's tokens, tool text included, stay within the KV it
reserved at intake.

The ``python`` handler runs each call as ``python -I`` in a process of its own, killed on a
timeout, with the CPU time and memory limited. That is no sandbox: it has no
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CONTINUES, FINAL_PREFILL, TOOLS, TURN_IDS, CurrentForwardPassInfo, WalkStep
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
TOOL_CALLS = "tool_calls"
TOOL_TEXT = "tool_text"
# the exits of the router: a turn with no tool call answers (the plan ends); one with calls runs them
ANSWER = "answer"
TOOLS_EXIT = "tools"

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
        """The text before and after the results (each `wrap`ped and tokenized apart): what ends the
        turn and opens the user's, then what closes it and opens the assistant's next."""
        opening = "<|im_end|>\n<|im_start|>user"
        think = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        closing = "<|im_end|>\n<|im_start|>assistant\n" + think
        return opening, closing

    @staticmethod
    def wrap(result: str) -> str:
        # the template trims a message's content
        return f"\n<tool_response>\n{result.strip()}\n</tool_response>"


def run_python(
    code: str, timeout: float = 10.0, cpu_seconds: int = 10, memory_bytes: int = 1 << 30, max_chars: int = 4000,
) -> str:
    """``code`` run by ``python -I`` in a process group of its own; its output (stdout and stderr), or
    an error naming the limit it hit."""

    # the limits are set by the child itself, before it runs the code (no fork hook in a threaded process)
    wrapper = (
        "import resource, sys\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({memory_bytes}, {memory_bytes}))\n"
        "exec(compile(sys.argv[1], '<tool>', 'exec'))\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", wrapper, code], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True,
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


SPEAK = "speak"
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


def _speak(arguments: Mapping[str, str]) -> str:
    # the audio is not made here: the Tool node sends the text on to the speech model (`ToolSubmodule`)
    return "Spoken." if arguments.get("text", "").strip() else "Error: speak takes the argument text."


# The tools a serving config's ``tools`` can name: each its handler (arguments to the text the model
# reads back) and the schema the chat template shows the model.
TOOL_HANDLERS: dict[str, Callable[[Mapping[str, str]], str]] = {"python": _python, SPEAK: _speak}
TOOL_SCHEMAS = {"python": PYTHON_TOOL_SCHEMA, SPEAK: SPEAK_SCHEMA}


def check_tools(names: Sequence[str]) -> tuple[str, ...]:
    if unknown := sorted(set(names) - set(TOOL_HANDLERS)):
        raise ValueError(f"unknown tools {unknown}; the tools are {sorted(TOOL_HANDLERS)}")
    return tuple(names)


def tool_walks() -> dict[str, GraphSection]:
    """The ``route`` walk (the turn's ids in, the parsed calls out) and the ``tool`` walk (the calls
    in, ``tool_text`` out, persisted and counted toward the request's output budget)."""
    return {
        ROUTE_WALK: GraphNode(
            name=ROUTER_NODE, input_names=["new_token", TURN_IDS],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=TOOL_CALLS, persist=True)],
        ),
        TOOL_WALK: GraphNode(
            name=TOOL_NODE, input_names=[TOOL_CALLS],
            outputs=[GraphEdge(
                next_node=EMPTY_DESTINATION, name=TOOL_TEXT, persist=True, conductor_new_token=True)],
        ),
    }


def tool_plan(
    prefills: list[WalkStep], decode_walk: str, prefill_walk: str, enable_thinking: bool, speak: bool = False,
) -> list[WalkStep]:
    """The plan's own ``prefills`` (one, or a spoken prompt's several), decode, the router, the tool, and
    the prefill that continues the reply and comes back to decode; the router's ``answer`` ends the plan."""
    n = len(prefills)
    return [
        *(replace(step, step_metadata={**step.step_metadata, TOOLS: True}) for step in prefills),
        WalkStep(decode_walk, bind={"text_inputs": "new_token"}, step_metadata={TOOLS: True}),
        WalkStep(ROUTE_WALK, exits={ANSWER: n + 4, TOOLS_EXIT: n + 2}),
        # `speak`: whether the request has a speech output the tool's `speak` text may go to
        WalkStep(TOOL_WALK, step_metadata={"enable_thinking": enable_thinking, SPEAK: speak}),
        WalkStep(
            prefill_walk, bind={"text_inputs": TOOL_TEXT}, exits={"": n},
            step_metadata={FINAL_PREFILL: True, CONTINUES: True, TOOLS: True},
        ),
    ]


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

    def __init__(self, tokenizer, device: str | torch.device = "cpu", fmt: QwenToolFormat | None = None):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.format = fmt or QwenToolFormat()

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        first = inputs["new_token"][0].reshape(-1)
        ids = torch.cat([first, *(t.reshape(-1) for t in inputs.get(TURN_IDS, []))]).cpu()
        return ARNodeInputs(input_seq_len=0, kwargs={"ids": ids})

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows=(), **kwargs,
    ) -> dict[str, NameToTensorList]:
        out = {}
        for rid, row in zip(engine_inputs.request_ids, rows, strict=True):
            ids = row["ids"]
            calls = self.format.parse(self.tokenizer.decode(ids.tolist(), skip_special_tokens=False))
            payload = {"turn_len": ids.numel(), "calls": [[c.name, dict(c.arguments)] for c in calls]}
            out[rid] = {TOOL_CALLS: [self._tensor(_pack(payload), torch.uint8)]}
        return out

    def check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo, outputs: Mapping[str, list[torch.Tensor]],
    ) -> StopCheck:
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
            SPEAK: bool(fwd_info.step_metadata.get(SPEAK, False)),
        })

    def _text_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _run(self, name: str, arguments: Mapping[str, str], speaks: bool) -> str:
        handler = self.handlers.get(name)
        if handler is None:
            return f"Error: there is no tool {name!r}; the tools are {sorted(self.handlers)}."
        if name == SPEAK and not speaks:
            return "Error: this request has no spoken output; answer in text."
        try:
            return handler(arguments)
        except Exception as error:  # the model reads it; the request goes on
            return f"Error: {name} failed: {type(error).__name__}: {error}"

    def _text(self, turn_len: int, results: list[str], enable_thinking: bool, budget: int) -> list[int]:
        """``tool_text``: the turn's length, then the ids to write (the template's tail)."""
        opening, closing = self.format.render_frame(enable_thinking)
        head, tail = self._text_ids(opening), self._text_ids(closing)
        # what is left of the request's budget bounds the text; the results give way, evenly
        room = budget - 1 - len(head) - len(tail)
        if room < 0 or not results:
            return [turn_len]
        share = room // len(results)
        body = [t for r in results for t in self._text_ids(self.format.wrap(r))[:share]]
        return [turn_len, *head, *body, *tail]

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows=(), **kwargs,
    ) -> dict[str, NameToTensorList]:
        # every call of every row at once; a row's results come back together
        futures = [[self._pool.submit(self._run, name, args, row[SPEAK]) for name, args in row["payload"]["calls"]]
                   for row in rows]
        out = {}
        for rid, row, row_futures in zip(engine_inputs.request_ids, rows, futures, strict=True):
            results = [f.result() for f in row_futures]
            out[rid] = {TOOL_TEXT: [self._tensor(
                self._text(row["payload"]["turn_len"], results, row["enable_thinking"], row["budget"]),
                torch.long)]}
            # what the turn says aloud goes on to the speech model, one item; a row that says nothing
            # routes no tensor on the stream edge, so that model hears nothing of it
            said = [args for (name, args), result in zip(row["payload"]["calls"], results, strict=True)
                    if name == SPEAK and not result.startswith("Error")]
            if said:
                text = " ".join(args["text"].strip() for args in said)
                out[rid][SPEAK_TEXT] = [self._tensor(_pack({"text": text, "voice": said[0].get("voice")}), torch.uint8)]
        return out
