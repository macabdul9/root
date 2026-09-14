"""Run root's agent loop against a shell it does not own.

The interactive terminal and `root-eval` give the model tools that act on the
local filesystem. A benchmark like Terminal-Bench-Science instead hands the
agent a disposable container and grades what is left in it, so the tools have to
execute over there. Everything here is about that one difference: a `run_bash`
whose body is a callable supplied by whoever owns the container.

No dependency on any particular harness. `solve()` takes a function that runs a
command somewhere and returns what it printed; `root.harbor` supplies one backed
by Harbor's environment, and the tests supply one backed by a dictionary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .agent import AgentResult, run_agent
from .config import AgentSpec
from .tools import Parameter, Tool

logger = logging.getLogger(__name__)

# Far above the 800 characters the local tools truncate at. That limit suits a
# chat turn quoting one grep hit; a terminal task reads directory listings and
# tracebacks, and cutting those off is indistinguishable to the model from the
# command not having worked.
MAX_OUTPUT_CHARS = 8000


@dataclass(frozen=True, slots=True)
class Command:
    """What one shell command printed, and whether it worked."""

    stdout: str
    stderr: str
    return_code: int

    def report(self, max_chars: int = MAX_OUTPUT_CHARS) -> str:
        """The single string the model sees for this command.

        stderr is included rather than dropped: a task fails on a traceback far
        more often than on silence, and the model cannot ask for it separately.
        """
        body = "\n".join(part for part in (self.stdout.strip(), self.stderr.strip()) if part)
        if len(body) > max_chars:
            body = body[:max_chars] + "\n[truncated]"
        if self.return_code != 0:
            return f"exit {self.return_code}\n{body}".strip()
        return body or "exit 0, no output"


Run = Callable[[str], Command]


@dataclass(frozen=True, slots=True)
class Attempt:
    """One pass at a task: the loop's result, and what it actually ran.

    The commands are recorded at the tool boundary rather than read back off
    `AgentResult.steps`, because a step's `argument` is the rendered form shown
    in the trace - quoted for display - and a transcript wants the string that
    reached the shell.
    """

    result: AgentResult
    commands: tuple[tuple[str, str], ...]


def bash_tool(
    run: Run,
    max_chars: int = MAX_OUTPUT_CHARS,
    log: list[tuple[str, str]] | None = None,
) -> Tool:
    """A `run_bash` that executes wherever `run` sends it.

    Deliberately without the refusal list `root.tools.run_bash` carries. That
    list protects the machine root is running on; here the command lands in a
    container built to be thrown away, and the task legitimately needs to write
    files, install packages and delete things. Refusing those would cap the
    score for a reason that has nothing to do with the model.
    """

    def execute(command: str) -> str:
        output = run(command).report(max_chars)
        if log is not None:
            log.append((command, output))
        return output

    return Tool(
        name="run_bash",
        description="Run a shell command in the task environment and return its output.",
        parameters=(
            Parameter(
                "command",
                "A shell command, such as ls -la /app or python script.py.",
                freeform=True,
            ),
        ),
        run=execute,
    )


def solve(
    model,
    spec: AgentSpec,
    instruction: str,
    run: Run,
    max_chars: int = MAX_OUTPUT_CHARS,
) -> Attempt:
    """Work through one task instruction, one shell command at a time."""
    log: list[tuple[str, str]] = []
    tool = bash_tool(run, max_chars, log)
    result = run_agent(model, spec, instruction, {"run_bash": tool})
    logger.info(
        "finished agent=%s steps=%d commands=%d step_limit=%s",
        spec.name,
        len(result.steps),
        len(log),
        result.hit_step_limit,
    )
    return Attempt(result=result, commands=tuple(log))


def transcript(attempt: Attempt) -> str:
    """Every command the agent ran and what came back.

    The one artifact worth having when a trial scores zero: a step count says
    the loop ran, and only the commands say whether the model misread the task,
    wrote to the wrong path, or reported success without doing anything.
    """
    lines = []
    for command, output in attempt.commands:
        lines.append(f"$ {command}")
        lines.append(output)
    lines.append(f"[answer] {attempt.result.answer}")
    return "\n".join(lines) + "\n"


def usage(result: AgentResult) -> dict[str, int]:
    """Token counts across the whole loop.

    Summed over steps rather than taken from the last one: each step re-sends
    the conversation, so the prompt tokens of a ten-step task are most of what
    the model actually processed.
    """
    return {
        "n_input_tokens": sum(step.prompt_tokens for step in result.steps),
        "n_output_tokens": sum(step.generated_tokens for step in result.steps),
    }
