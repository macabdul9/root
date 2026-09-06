from __future__ import annotations

import sys

from .agent import AgentResult, Step

LEVELS = ("off", "on", "full")

RESULT_PREVIEW_CHARS = 200
ARGUMENT_PREVIEW_CHARS = 120

_DIM = "\033[2m"
_RESET = "\033[0m"


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
        lines.append(f"  ● {step.call.name}({argument!r})")
        lines.append(f"    ⎿ {_flatten(step.call.result, RESULT_PREVIEW_CHARS)}")
        lines.append(_dim(f"      {meta} · tool {step.tool_seconds:.1f}s{forced}", color))
    elif level == "full":
        lines.append(_dim(f"  ● answer · {meta}", color))

    if level == "full":
        raw = _flatten(step.generation, RESULT_PREVIEW_CHARS)
        lines.append(_dim(f"      raw: {raw}", color))

    return lines


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
