from __future__ import annotations

import ast
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import AgentSpec
from .tools import Tool

if TYPE_CHECKING:
    from .backend import LocalModel

logger = logging.getLogger(__name__)

# The markers the LFM2 chat template trains the model to emit around a call.
TOOL_CALL_START = "<|tool_call_start|>["
TOOL_CALL_END = "<|tool_call_end|>"

TOOL_NUDGE = (
    "Call a tool when it helps. Never write the tool result yourself: stop after "
    "the call and wait for it. Once you have the result, answer in plain text."
)

_MARKED_CALL = re.compile(
    r"<\|tool_call_start\|>\s*\[?(.+?)\]?\s*(?:<\|tool_call_end\|>|$)", re.DOTALL
)
# Models that lose the markers still tend to emit a bare call on its own line.
_BARE_CALL = re.compile(r"^\s*([a-zA-Z_]\w*\(.*\))\s*$", re.MULTILINE)
_LOOSE_CALL = re.compile(r"([a-zA-Z_]\w*)\((.*)\)", re.DOTALL)
_KEYWORD_PREFIX = re.compile(r"^[a-zA-Z_]\w*\s*=\s*")


@dataclass(slots=True)
class ToolCall:
    name: str
    argument: str
    result: str


@dataclass(slots=True)
class Step:
    """One pass through the loop: a generation, and the call it asked for."""

    index: int
    generation: str
    seconds: float
    prompt_tokens: int
    generated_tokens: int
    call: ToolCall | None = None
    tool_seconds: float = 0.0
    forced: bool = False


@dataclass(slots=True)
class AgentResult:
    agent: str
    prompt: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    hit_step_limit: bool = False

    @property
    def calls(self) -> list[ToolCall]:
        return [step.call for step in self.steps if step.call]

    @property
    def seconds(self) -> float:
        return sum(step.seconds + step.tool_seconds for step in self.steps)

    @property
    def generated_tokens(self) -> int:
        return sum(step.generated_tokens for step in self.steps)


def _first_string_argument(call: ast.Call) -> str:
    for node in [kw.value for kw in call.keywords] + list(call.args):
        if isinstance(node, ast.Constant):
            return str(node.value)
    return ""


def _parse_call_source(source: str) -> tuple[str, str] | None:
    try:
        node = ast.parse(source.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    return node.func.id, _first_string_argument(node)


def _parse_call_loosely(source: str) -> tuple[str, str] | None:
    """Recover a call whose argument is not valid Python.

    A tool that carries code hits this constantly: the model writes
    `run_python(code="print(len("x"))")` and the nested quotes make the whole
    call unparseable, even though what it meant is unambiguous.
    """
    match = _LOOSE_CALL.fullmatch(source.strip())
    if match is None:
        return None
    argument = _KEYWORD_PREFIX.sub("", match.group(2).strip())
    if len(argument) >= 2 and argument[0] == argument[-1] and argument[0] in "\"'":
        argument = argument[1:-1]
    argument = argument.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
    return match.group(1), argument


def parse_tool_call(text: str, tool_names: set[str]) -> tuple[str, str] | None:
    """Pull a single tool call out of a generation.

    Prefers the tool-call markers the chat template trains the model to emit and
    falls back to a bare `name(arg)` line, which is what small models produce
    once they drop the markers.
    """
    marked = _MARKED_CALL.search(text)
    if marked:
        parsed = _parse_call_source(marked.group(1))
        if parsed:
            return parsed
        loose = _parse_call_loosely(marked.group(1))
        if loose and loose[0] in tool_names:
            return loose
    for match in _BARE_CALL.finditer(text):
        parsed = _parse_call_source(match.group(1)) or _parse_call_loosely(match.group(1))
        if parsed and parsed[0] in tool_names:
            return parsed
    return None


def _strip_after_call(text: str) -> str:
    """Cut anything generated past the tool call.

    Small models like to invent the tool's output and keep talking; only the
    call itself should go back into the conversation.
    """
    end = text.find(TOOL_CALL_END)
    if end == -1:
        return text.strip()
    return text[: end + len(TOOL_CALL_END)].strip()


def run_agent(
    model: LocalModel,
    spec: AgentSpec,
    prompt: str,
    tools: dict[str, Tool] | None = None,
    history: list[dict[str, str]] | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    """Answer one prompt, calling tools until the model stops asking for them.

    `history` carries earlier turns of a conversation, and `on_step` fires as
    each step finishes so a caller can trace the loop while it runs instead of
    waiting for the whole thing.
    """
    available = {name: tools[name] for name in spec.tools} if tools else {}
    system = spec.system_prompt.strip()
    if available:
        system = f"{system}\n{TOOL_NUDGE}"

    schemas = [tool.schema() for tool in available.values()] or None
    messages = [
        {"role": "system", "content": system},
        *(history or []),
        {"role": "user", "content": prompt},
    ]
    result = AgentResult(agent=spec.name, prompt=prompt, answer="", messages=messages)

    for index in range(spec.max_steps):
        # A 350M model often answers from memory instead of reaching for a tool;
        # opening its turn with the call marker removes that option on step 0.
        # Forcing the same on a retry is worse: it spends every remaining step
        # on variants of the code that just failed.
        prefill = TOOL_CALL_START if available and spec.force_first_call and index == 0 else None
        generation = model.generate(
            messages,
            tools=schemas,
            max_new_tokens=spec.max_new_tokens,
            temperature=spec.temperature,
            stop=[TOOL_CALL_END] if available else None,
            prefill=prefill,
        )
        step = Step(
            index=index,
            generation=generation.text,
            seconds=generation.seconds,
            prompt_tokens=generation.prompt_tokens,
            generated_tokens=generation.generated_tokens,
            forced=prefill is not None,
        )
        text = generation.text

        call = parse_tool_call(text, set(available)) if available else None
        if call is None:
            result.answer = text.strip()
            result.steps.append(step)
            if on_step:
                on_step(step)
            return result

        name, argument = call
        messages.append({"role": "assistant", "content": _strip_after_call(text)})
        started = time.monotonic()
        if name not in available:
            observation = f"error: no tool named {name!r}; available: {sorted(available)}"
        else:
            try:
                observation = available[name].run(argument)
            except Exception as exc:
                # The model is the retry mechanism here: it reads the error and tries again.
                observation = f"error: {exc}"
        step.tool_seconds = time.monotonic() - started
        step.call = ToolCall(name=name, argument=argument, result=observation)
        # The trace is the user-facing view of this; keep the log for debugging.
        logger.debug("step=%d tool=%s argument=%r", index, name, argument)
        result.steps.append(step)
        if on_step:
            on_step(step)

        # Left alone after a failure the model answers from imagination rather
        # than fixing the call, so say what to do with the error.
        content = observation
        if observation.startswith("error:") and index + 1 < spec.max_steps:
            content = f"{observation}\nFix the call and try again."
        messages.append({"role": "tool", "content": content})

    result.hit_step_limit = True
    final = model.generate(
        messages + [{"role": "user", "content": "Answer now, without calling a tool."}],
        max_new_tokens=spec.max_new_tokens,
        temperature=spec.temperature,
    )
    result.answer = final.text.strip()
    last = Step(
        index=spec.max_steps,
        generation=final.text,
        seconds=final.seconds,
        prompt_tokens=final.prompt_tokens,
        generated_tokens=final.generated_tokens,
    )
    result.steps.append(last)
    if on_step:
        on_step(last)
    return result
