from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .config import AgentSpec
from .formats import GENERIC, CallFormat, strip_calls, strip_thinking
from .identity import describe
from .route import is_conversational, is_identity_question
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
    # Answered without asking the model, so there is no timing, no token count
    # and no advice about agents to give.
    from_template: bool = False

    @property
    def calls(self) -> list[ToolCall]:
        return [step.call for step in self.steps if step.call]

    @property
    def seconds(self) -> float:
        return sum(step.seconds + step.tool_seconds for step in self.steps)

    @property
    def generated_tokens(self) -> int:
        return sum(step.generated_tokens for step in self.steps)


NOTHING_USABLE = "the model wrote no answer; /trace full shows what it produced"


def _answer_from(text: str, call_format: CallFormat, calls: list[ToolCall]) -> str:
    """The answer, with reasoning and tool markup taken out.

    Stripping can empty the text entirely: a small model often ends a turn by
    repeating the call rather than writing prose about it. The tool already has
    the answer in that case, so show what it returned instead of nothing.
    """
    answer = strip_calls(strip_thinking(text), call_format)
    if answer:
        return answer
    if calls:
        return calls[-1].result
    return NOTHING_USABLE if text.strip() else ""


def compose_system(spec: AgentSpec, available: dict[str, Tool], model_id: str) -> str:
    """The agent's prompt, behind the preamble every agent shares.

    Substitution is plain replacement rather than str.format, because prompts
    contain braces of their own and a JSON example should not be a template.
    """
    if not spec.preamble:
        return spec.system_prompt.strip()
    filled = (
        spec.preamble.replace("{{model}}", model_id)
        .replace("{{agent}}", spec.name)
        .replace("{{tools}}", ", ".join(available) or "none")
        .replace("{{date}}", datetime.now().strftime("%Y-%m-%d"))
    )
    return f"{filled.strip()}\n\n{spec.system_prompt.strip()}"


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
    on_token: Callable[[str], None] | None = None,
    agents: dict[str, AgentSpec] | None = None,
) -> AgentResult:
    """Answer one prompt, calling tools until the model stops asking for them.

    `history` carries earlier turns of a conversation, `on_step` fires as each
    step finishes so a caller can trace the loop while it runs, and `on_token`
    receives text as the model produces it.

    A question about what root is never reaches the model. Asked who it is, a
    model this size reports having been built by Naver, or by Microsoft, or by
    whoever its training data suggests; the answer is assembled from what is
    actually loaded instead.
    """
    if is_identity_question(prompt):
        answer = describe(getattr(model, "model_id", "an unknown model"), tools or {}, agents or {})
        if on_token:
            on_token(answer)
        return AgentResult(agent=spec.name, prompt=prompt, answer=answer, from_template=True)
    call_format = getattr(model, "call_format", GENERIC)
    # A tool the agent names but this session does not have is dropped rather
    # than fatal: an MCP server that is configured and switched off leaves the
    # agent with nothing to call, and it should still answer.
    available = {name: tools[name] for name in spec.tools if name in tools} if tools else {}
    system = compose_system(spec, available, getattr(model, "model_id", "an unknown model"))
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
            decoding=spec.decoding,
            stop=list(call_format.stop) if available else None,
            prefill=prefill,
            # A forced turn is a tool call by construction: nothing worth
            # streaming, and the one case where the output can be constrained
            # to a well-formed call without ruling out a plain answer.
            on_token=None if prefill else on_token,
            constrain=prefill is not None,
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
            result.answer = _answer_from(text, call_format, result.calls)
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
        decoding=spec.decoding,
        on_token=on_token,
    )
    result.answer = _answer_from(final.text, call_format, result.calls)
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
