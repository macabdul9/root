from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .agent import AgentResult

NO_TOOL = "none"


@dataclass(frozen=True, slots=True)
class EvalCase:
    agent: str
    prompt: str
    workspace: str | None = None
    expect_tool: str | None = None
    expect_argument_contains: str | None = None
    expect_answer_contains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError(f"case for agent {self.agent}: empty prompt")
        if self.expect_argument_contains and self.expect_tool in (None, NO_TOOL):
            raise ValueError(
                f"case for agent {self.agent}: expect_argument_contains needs an expect_tool"
            )


@dataclass(slots=True)
class CaseScore:
    case: EvalCase
    tool_ok: bool
    argument_ok: bool
    answer_ok: bool
    tools_called: list[str]
    answer: str

    @property
    def passed(self) -> bool:
        return self.tool_ok and self.argument_ok and self.answer_ok

    def failures(self) -> list[str]:
        reasons = []
        if not self.tool_ok:
            reasons.append(
                f"expected tool {self.case.expect_tool}, called {self.tools_called or 'nothing'}"
            )
        if not self.argument_ok:
            reasons.append(f"argument missing {self.case.expect_argument_contains!r}")
        if not self.answer_ok:
            missing = [
                s for s in self.case.expect_answer_contains if s.lower() not in self.answer.lower()
            ]
            reasons.append(f"answer missing {missing}")
        return reasons


def load_cases(path: Path) -> list[EvalCase]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_cases = document.get("cases") or []
    if not raw_cases:
        raise ValueError(f"no cases defined in {path}")
    cases = []
    for entry in raw_cases:
        expected = entry.get("expect_answer_contains") or ()
        if isinstance(expected, str):
            expected = (expected,)
        cases.append(
            EvalCase(
                agent=entry["agent"],
                prompt=entry["prompt"],
                workspace=entry.get("workspace"),
                expect_tool=entry.get("expect_tool"),
                expect_argument_contains=entry.get("expect_argument_contains"),
                expect_answer_contains=tuple(str(s) for s in expected),
            )
        )
    return cases


def score_case(case: EvalCase, result: AgentResult) -> CaseScore:
    """Grade one run.

    An omitted expectation is not a check: only `expect_tool: none` asserts that
    the agent answered without reaching for a tool.
    """
    called = [call.name for call in result.calls]

    if case.expect_tool is None:
        tool_ok = True
    elif case.expect_tool == NO_TOOL:
        tool_ok = not called
    else:
        tool_ok = case.expect_tool in called

    argument_ok = True
    if case.expect_argument_contains:
        wanted = case.expect_argument_contains.lower()
        argument_ok = any(
            call.name == case.expect_tool and wanted in call.argument.lower()
            for call in result.calls
        )

    answer = result.answer.lower()
    answer_ok = all(s.lower() in answer for s in case.expect_answer_contains)

    return CaseScore(
        case=case,
        tool_ok=tool_ok,
        argument_ok=argument_ok,
        answer_ok=answer_ok,
        tools_called=called,
        answer=result.answer,
    )


def accuracy(scores: list[CaseScore]) -> dict[str, float]:
    if not scores:
        return {"tool": 0.0, "answer": 0.0, "overall": 0.0}
    return {
        "tool": sum(s.tool_ok and s.argument_ok for s in scores) / len(scores),
        "answer": sum(s.answer_ok for s in scores) / len(scores),
        "overall": sum(s.passed for s in scores) / len(scores),
    }
