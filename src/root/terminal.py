from __future__ import annotations

import logging
import readline
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .agent import AgentResult, Step, run_agent
from .config import AgentSpec
from .route import AUTO, RULES, choose_agent
from .tools import Tool, build_tools
from .trace import LEVELS, render_route, render_step, render_summary, render_turn, use_color

if TYPE_CHECKING:
    from .backend import LocalModel

logger = logging.getLogger(__name__)

HISTORY_FILE = Path.home() / ".root_history"
HISTORY_TURNS = 4

BANNER = "root · {model} · {agent} · /help for commands, ctrl-d to leave"
NO_TOOL_HINT = "  no rule matched, and {agent} has no tools - try /agent python"

HELP = """\
/agents            list configured agents
/agent NAME        switch agent, clearing the conversation ("auto" to route)
/tools             show the tools the current agent may call
/routes            show the rules auto uses to pick an agent
/trace [LEVEL]     show or set the trace level: off, on, full
/last              replay the previous turn in full detail
/workspace [PATH]  show or change where file tools look
/new               forget the conversation so far
/help              this list
/quit              leave"""


@dataclass(slots=True)
class Session:
    agents: dict[str, AgentSpec]
    agent: str
    tools: dict[str, Tool]
    workspace: Path
    history: list[dict[str, str]] = field(default_factory=list)
    running: bool = True
    trace: str = "on"
    last: AgentResult | None = None
    last_spec: AgentSpec | None = None

    @property
    def spec(self) -> AgentSpec:
        """The agent a prompt would run under, ignoring routing."""
        return self.agents[self.agent if self.agent != AUTO else "chat"]

    def resolve(self, prompt: str) -> tuple[AgentSpec, str | None]:
        if self.agent != AUTO:
            return self.agents[self.agent], None
        route = choose_agent(prompt, set(self.agents))
        return self.agents[route.agent], route.rule

    def remember(self, prompt: str, answer: str) -> None:
        """Keep the exchange, but not the tool calls that produced it.

        Replaying old calls into a 350M model's context costs tokens and invites
        it to answer from a stale observation instead of the current question.
        """
        self.history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        del self.history[: max(0, len(self.history) - HISTORY_TURNS * 2)]


def run_command(session: Session, line: str) -> str:
    name, _, argument = line[1:].partition(" ")
    argument = argument.strip()

    if name in {"quit", "exit", "q"}:
        session.running = False
        return ""
    if name == "help":
        return HELP
    if name == "agents":
        rows = [f"{'*' if session.agent == AUTO else ' '} {AUTO:9s} route by the prompt"]
        rows += [
            f"{'*' if key == session.agent else ' '} {key:9s} {list(spec.tools) or '-'}"
            for key, spec in session.agents.items()
        ]
        return "\n".join(rows)
    if name == "agent":
        if argument != AUTO and argument not in session.agents:
            return f"unknown agent {argument!r}; try {[AUTO, *sorted(session.agents)]}"
        session.agent = argument
        session.history.clear()
        return f"agent: {argument}"
    if name == "routes":
        return "\n".join(f"{agent:8s} {rule}" for rule, _, agent in RULES)
    if name == "tools":
        if session.agent == AUTO:
            return "auto picks an agent per prompt; /routes shows the rules"
        used = [session.tools[t] for t in session.spec.tools]
        if not used:
            return f"{session.agent} answers without tools"
        return "\n".join(f"{t.name}({t.parameter}) - {t.description}" for t in used)
    if name == "workspace":
        if argument:
            path = Path(argument).expanduser()
            if not path.is_dir():
                return f"not a directory: {argument}"
            session.workspace = path.resolve()
            session.tools = build_tools(session.workspace)
        return f"workspace: {session.workspace}"
    if name == "trace":
        if argument:
            if argument not in LEVELS:
                return f"trace level must be one of {list(LEVELS)}"
            session.trace = argument
        return f"trace: {session.trace}"
    if name == "last":
        if session.last is None:
            return "nothing to replay yet"
        lines = render_turn(session.last, level="full", color=use_color())
        return "\n".join([*lines, session.last.answer])
    if name == "new":
        session.history.clear()
        return "conversation cleared"
    return f"unknown command {line.split()[0]!r}; /help lists them"


def answer(model: LocalModel, session: Session, prompt: str) -> AgentResult:
    color = use_color()
    spec, rule = session.resolve(prompt)

    if session.agent == AUTO and session.trace != "off":
        print(render_route(spec.name, rule, color))

    def show(step: Step) -> None:
        for line in render_step(step, session.trace, color):
            print(line)

    result = run_agent(
        model,
        spec,
        prompt,
        session.tools,
        history=session.history,
        on_step=show,
    )
    session.remember(prompt, result.answer)
    session.last = result
    session.last_spec = spec
    return result


def start_terminal(model: LocalModel, session: Session) -> int:
    """Read prompts until end of input, keeping the model loaded between them."""
    # Loading the model logs at INFO; nothing below that belongs in a session.
    logging.getLogger().setLevel(logging.WARNING)
    if HISTORY_FILE.exists():
        readline.read_history_file(HISTORY_FILE)

    print(BANNER.format(model=model.model_id.split("/")[-1], agent=session.agent))
    while session.running:
        try:
            line = input(f"{session.agent}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            message = run_command(session, line)
            if message:
                print(message)
            continue

        try:
            result = answer(model, session, line)
        except KeyboardInterrupt:
            print("\ninterrupted")
            continue
        spec = session.last_spec or session.spec
        print(result.answer)
        if session.trace != "off":
            print(render_summary(result, len(spec.tools), color=use_color()))
            if session.agent == AUTO and not spec.tools:
                print(NO_TOOL_HINT.format(agent=spec.name))

    readline.write_history_file(HISTORY_FILE)
    return 0
