from __future__ import annotations

import re
import sys
from collections.abc import Callable

from .agent import AgentResult, Step
from .formats import THINK_CLOSE, THINK_OPEN

LEVELS = ("off", "on", "full")

RESULT_PREVIEW_CHARS = 200
TAG_TAIL = 32
CODE_INDENT = "    "
ARGUMENT_PREVIEW_CHARS = 120

_DIM = "\033[2m"
_BOLD = "\033[1m"
_CODE = "\033[36m"
_RESET = "\033[0m"

_BOLD_SPAN = re.compile(r"\*\*(.+?)\*\*")
_INLINE_CODE = re.compile(r"`([^`]+)`")


def _dim(text: str, color: bool) -> str:
    return f"{_DIM}{text}{_RESET}" if color else text


def _flatten(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def use_color() -> bool:
    return sys.stdout.isatty()


def render_step(step: Step, level: str = "on", color: bool = False) -> list[str]:
    """One step as terminal lines: what was called, what came back, what it cost."""
    if level == "off":
        return []

    meta = f"{step.seconds:.1f}s · {step.generated_tokens} tok"
    lines = []

    if step.call:
        argument = _flatten(step.call.argument, ARGUMENT_PREVIEW_CHARS)
        forced = " · forced" if step.forced else ""
        lines.append(f"  ● {step.call.name}({argument})")
        lines.append(f"    ⎿ {_flatten(step.call.result, RESULT_PREVIEW_CHARS)}")
        lines.append(_dim(f"      {meta} · tool {step.tool_seconds:.1f}s{forced}", color))
    elif level == "full":
        lines.append(_dim(f"  ● answer · {meta}", color))

    if level == "full":
        raw = _flatten(step.generation, RESULT_PREVIEW_CHARS)
        lines.append(_dim(f"      raw: {raw}", color))

    return lines


def render_route(agent: str, rule: str | None, color: bool = False) -> str:
    why = rule or "no rule matched"
    return _dim(f"  ● route → {agent} ({why})", color)


def render_tier(tier: str, model_id: str, score: float | None, color: bool = False) -> str:
    effort = "" if score is None else f", effort {score:.2f}"
    return _dim(f"  ● tier → {tier} ({model_id.split('/')[-1]}{effort})", color)


class StreamGate:
    """Forward streamed text, without leaking a tool call to the screen.

    A generation is only known to be an answer once it is clear it is not a
    call, so text that could still become the format's opening marker is held
    back, and everything is dropped once the marker actually arrives.
    """

    def __init__(self, marker: str, write: Callable[[str], None], thinking: bool = False) -> None:
        self.marker = marker
        self.write = write
        self.buffer = ""
        self.suppressed = False
        self.emitted = False
        self.thinking = thinking

    def feed(self, chunk: str) -> None:
        if self.suppressed:
            return
        self.buffer += chunk
        if self.thinking and not self._leave_thinking():
            return
        if self._enter_thinking():
            return
        if self.marker and self.marker in self.buffer:
            self.suppressed = True
            self.buffer = ""
            return
        held = self._held()
        ready, self.buffer = (
            self.buffer[: len(self.buffer) - held],
            self.buffer[len(self.buffer) - held :],
        )
        if ready:
            self.emitted = True
            self.write(ready)

    def _leave_thinking(self) -> bool:
        """Drop reasoning until its closing tag, then resume."""
        closing = THINK_CLOSE.search(self.buffer)
        if closing is None:
            # Keep only enough tail to match a tag split across chunks.
            self.buffer = self.buffer[-TAG_TAIL:]
            return False
        self.thinking = False
        self.buffer = self.buffer[closing.end() :]
        return True

    def _enter_thinking(self) -> bool:
        """Emit anything before a reasoning block, then start dropping."""
        opening = THINK_OPEN.search(self.buffer)
        if opening is None:
            return False
        before = self.buffer[: opening.start()]
        if before:
            self.emitted = True
            self.write(before)
        self.thinking = True
        self.buffer = self.buffer[opening.end() :]
        return not self._leave_thinking()

    def close(self) -> None:
        if self.thinking:
            self.buffer = ""
        if not self.suppressed and self.buffer:
            self.emitted = True
            self.write(self.buffer)
        self.buffer = ""

    def _held(self) -> int:
        """How much of the tail could still turn into the marker."""
        if not self.marker:
            return 0
        for size in range(min(len(self.buffer), len(self.marker) - 1), 0, -1):
            if self.marker.startswith(self.buffer[-size:]):
                return size
        return 0


def format_answer(text: str, color: bool = False) -> str:
    """Lay out a model answer for a terminal.

    Fenced blocks are indented and coloured so code is distinguishable from the
    prose around it, which matters here because these models mix the two freely.
    """
    lines = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            language = line.lstrip().removeprefix("```").strip()
            if in_code and language:
                lines.append(_dim(f"{CODE_INDENT}{language}", color))
            continue
        if in_code:
            lines.append(_colour(CODE_INDENT + line, _CODE, color))
        else:
            lines.append(_inline(line, color))
    return "\n".join(lines).strip("\n")


def _colour(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


def _inline(line: str, color: bool) -> str:
    if not color:
        return _BOLD_SPAN.sub(r"\1", _INLINE_CODE.sub(r"\1", line))
    line = _BOLD_SPAN.sub(f"{_BOLD}\\1{_RESET}", line)
    return _INLINE_CODE.sub(f"{_CODE}\\1{_RESET}", line)


def render_summary(result: AgentResult, tool_count: int, color: bool = False) -> str:
    calls = len(result.calls)
    parts = [
        f"{result.seconds:.1f}s",
        f"{result.generated_tokens} tok",
        f"{calls} call{'' if calls == 1 else 's'}",
    ]
    if not tool_count:
        parts.append("agent has no tools")
    if result.hit_step_limit:
        parts.append("step limit reached")
    return _dim(f"[{' · '.join(parts)}]", color)


def render_turn(result: AgentResult, level: str = "full", color: bool = False) -> list[str]:
    lines = []
    for step in result.steps:
        lines += render_step(step, level, color)
    return lines
