"""Keep working until a command says the task is done.

`run_agent` answers one prompt: it stops the moment the model produces a turn
without a tool call, whether or not anything was accomplished. That is right for
a question and wrong for a task. Editing a file until the tests pass needs a
second loop around the first, with something other than the model's own opinion
deciding when to stop.

The stopping condition is a shell command the user supplies, and its exit status
is the verdict. A model cannot talk its way past `pytest -q` returning 1, which
is the whole point: the check is the one part of the loop the model does not
control.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import AgentResult, run_agent
from .config import AgentSpec
from .tools import Tool

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SECONDS = 300
# Enough of a failing test run to act on. The tool budget is for a chat turn
# quoting one grep hit; a traceback cut to 800 characters loses the line that
# says what broke.
MAX_CHECK_OUTPUT_CHARS = 4000


@dataclass(frozen=True, slots=True)
class Outcome:
    passed: bool
    output: str


@dataclass(frozen=True, slots=True)
class Check:
    """The command whose exit status decides whether the work is finished."""

    command: str
    workspace: Path = Path(".")
    timeout_sec: int = CHECK_TIMEOUT_SECONDS

    def run(self) -> Outcome:
        try:
            finished = subprocess.run(
                ["bash", "-c", self.command],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            # A check that hangs is not a check that passed.
            return Outcome(False, f"the check did not finish in {self.timeout_sec}s")
        output = (finished.stdout + finished.stderr).strip()
        if len(output) > MAX_CHECK_OUTPUT_CHARS:
            output = output[-MAX_CHECK_OUTPUT_CHARS:]
        return Outcome(finished.returncode == 0, output)


@dataclass(slots=True)
class Pursuit:
    """What happened across every attempt at one task."""

    task: str
    command: str
    attempts: list[AgentResult] = field(default_factory=list)
    outcomes: list[Outcome] = field(default_factory=list)
    passed: bool = False
    already_passing: bool = False
    seconds: float = 0.0

    @property
    def calls(self) -> int:
        return sum(len(attempt.calls) for attempt in self.attempts)


def retry_prompt(task: str, outcome: Outcome) -> str:
    """What the model is told after a failed check.

    The output goes in verbatim because it is the only new information in the
    turn: without it the model rewrites the same thing, having no way to know
    what its last edit did.
    """
    return (
        "That did not work. The check still fails and printed this:\n"
        f"{outcome.output or '(no output)'}\n\n"
        f"Fix it, then stop. The task was: {task}"
    )


def pursue(
    model,
    spec: AgentSpec,
    task: str,
    tools: dict[str, Tool],
    check: Check,
    attempts: int = 5,
    deadline_sec: float | None = None,
    on_step=None,
    on_attempt=None,
) -> Pursuit:
    """Work at a task until the check passes, the attempts run out, or time does.

    The check runs first: a task already satisfied should cost nothing, and a
    check that cannot fail is a configuration mistake worth seeing immediately
    rather than after five attempts of an agent with nothing to do.
    """
    pursuit = Pursuit(task=task, command=check.command)
    started = time.monotonic()

    outcome = check.run()
    if outcome.passed:
        pursuit.passed = True
        pursuit.already_passing = True
        pursuit.outcomes.append(outcome)
        pursuit.seconds = time.monotonic() - started
        return pursuit

    history: list[dict[str, str]] = []
    prompt = task
    for attempt in range(attempts):
        result = run_agent(model, spec, prompt, tools, history=history, on_step=on_step)
        pursuit.attempts.append(result)

        outcome = check.run()
        pursuit.outcomes.append(outcome)
        logger.info(
            "attempt=%d calls=%d check=%s",
            attempt,
            len(result.calls),
            "passed" if outcome.passed else "failed",
        )
        if on_attempt:
            on_attempt(attempt, result, outcome)
        if outcome.passed:
            pursuit.passed = True
            break

        elapsed = time.monotonic() - started
        if deadline_sec is not None and elapsed >= deadline_sec:
            break
        # Carry the conversation forward, including the answer run_agent leaves
        # out of its own message list, so the next attempt is a continuation
        # rather than a fresh start on a workspace it no longer recognises.
        history = list(result.messages[1:])
        if result.answer:
            history.append({"role": "assistant", "content": result.answer})
        prompt = retry_prompt(task, outcome)

    pursuit.seconds = time.monotonic() - started
    return pursuit


def summarise(pursuit: Pursuit) -> str:
    """One line per attempt, and what the check said at the end."""
    if pursuit.already_passing:
        return f"`{pursuit.command}` already passes; nothing to do"
    lines = [
        f"attempt {index + 1}: {len(result.calls)} calls, check "
        f"{'passed' if outcome.passed else 'failed'}"
        for index, (result, outcome) in enumerate(
            zip(pursuit.attempts, pursuit.outcomes, strict=True)
        )
    ]
    verdict = "passed" if pursuit.passed else f"still failing after {len(pursuit.attempts)}"
    lines.append(f"`{pursuit.command}` {verdict} in {pursuit.seconds:.0f}s")
    return "\n".join(lines)
