from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import AgentSpec
from .formats import GENERIC, CallFormat, strip_thinking
from .route import is_conversational
from .tools import Tool

if TYPE_CHECKING:
    from .backend import LocalModel

logger = logging.getLogger(__name__)

TOOL_NUDGE = (
    "Call a tool when it helps. Never write the tool result yourself: stop after "
    "the call and wait for it. Once you have the result, answer in plain text."
)


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


def render_arguments(arguments: list[tuple[str | None, str]], tool: Tool | None = None) -> str:
    """Arguments as they should appear in a trace.

    A one-parameter tool shows just its quoted value, which is what nearly every
    call is; anything wider is named, so `write_file(path=..., content=...)`
    stays readable. Quoting happens here so callers print the result as-is.
    """
    single = (tool is not None and len(tool.parameters) == 1) or (
        len(arguments) == 1 and arguments[0][0] is None
    )
    if single:
        return repr(next((value for _, value in arguments), ""))
    return ", ".join(
        repr(value) if name is None else f"{name}={value!r}" for name, value in arguments
    )


def _strip_after_call(text: str, call_format: CallFormat) -> str:
    """Cut anything generated past the tool call.

    Small models like to invent the tool's output and keep talking; only the
    call itself should go back into the conversation.
    """
    for marker in call_format.stop:
        end = text.find(marker)
        if end != -1:
            return text[: end + len(marker)].strip()
    return text.strip()


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
    call_format = getattr(model, "call_format", GENERIC)
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
        # on variants of the code that just failed. A question about the
        # assistant itself is not a task, so nothing is forced there either.
        force = available and spec.force_first_call and index == 0 and not is_conversational(prompt)
        prefill = call_format.prefill if force and call_format.forceable else None
        generation = model.generate(
            messages,
            tools=schemas,
            max_new_tokens=spec.max_new_tokens,
            temperature=spec.temperature,
            stop=list(call_format.stop) if available else None,
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

        call = call_format.parse(text, set(available)) if available else None
        if call is None:
            result.answer = strip_thinking(text)
            result.steps.append(step)
            if on_step:
                on_step(step)
            return result

        name, arguments = call
        messages.append({"role": "assistant", "content": _strip_after_call(text, call_format)})
        started = time.monotonic()
        shown = render_arguments(arguments)
        if name not in available:
            observation = f"error: no tool named {name!r}; available: {sorted(available)}"
        else:
            try:
                tool = available[name]
                bound = tool.bind(arguments)
                shown = render_arguments([(key, value) for key, value in bound.items()], tool)
                observation = tool.run(**bound)
            except Exception as exc:
                # The model is the retry mechanism here: it reads the error and tries again.
                observation = f"error: {exc}"
        step.tool_seconds = time.monotonic() - started
        step.call = ToolCall(name=name, argument=shown, result=observation)
        # The trace is the user-facing view of this; keep the log for debugging.
        logger.debug("step=%d tool=%s arguments=%r", index, name, shown)
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
    result.answer = strip_thinking(final.text)
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
