from __future__ import annotations

import re
import sys

from .agent import AgentResult, Step

LEVELS = ("off", "on", "full")

RESULT_PREVIEW_CHARS = 200
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
