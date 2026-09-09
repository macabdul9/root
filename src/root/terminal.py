from __future__ import annotations

import logging
import readline
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .agent import AgentResult, Step, run_agent
from .config import (
    AgentSpec,
    ModelChoice,
    check_size,
    describe_hub_model,
    load_models,
    register_model,
    resolve_model,
    unregister_model,
)
from .engines import ENGINES, TRANSFORMERS, EngineUnavailable, load_engine
from .progress import working
from .route import AUTO, RULES, choose_agent
from .tools import Tool, build_tools
from .trace import (
    LEVELS,
    StreamGate,
    format_answer,
    render_route,
    render_step,
    render_summary,
    render_turn,
    use_color,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

HISTORY_FILE = Path.home() / ".root_history"
HISTORY_TURNS = 4

BANNER = "root · {model} · {agent} · /help for commands, ctrl-d to leave"
PROMPT = "root> "
NO_TOOL_HINT = "  no rule matched, and {agent} has no tools - try /agent python"
PINNED_HINT = "  pinned to {agent}; auto would have used {other} here - /agent auto"

HELP = """\
/agents            list configured agents
/agent NAME        switch agent, clearing the conversation ("auto" to route)
/model [NAME]      list models, or load one by alias or Hugging Face id
/model add ID      register a Hugging Face model so it stays in the list
/model remove NAME forget a registered model
/inference [NAME]  show or switch the engine: auto, transformers, vllm, sglang, ollama
                   (/engine does the same)
/tools             show the tools the current agent may call
/routes            show the rules auto uses to pick an agent
/trace [LEVEL]     show or set the trace level: off, on, full
/stream [on|off]   stream the answer as it is generated
/decoding [K=V]    show or set temperature, top_p, top_k, min_p, max_new_tokens
/last              replay the previous turn in full detail
/workspace [PATH]  show or change where file tools look
/new               forget the conversation so far
/help              this list
/quit              leave"""


def load_model(choice: ModelChoice, device: str | None, engine: str = TRANSFORMERS):
    """Bring a model up on one engine.

    Only the local engine takes trust_remote_code; a served engine loads the
    weights in its own process, where that is the server's decision.
    """
    return load_engine(choice.id, engine, device=device, trust_remote_code=choice.trust_remote_code)


@dataclass(slots=True)
class Session:
    agents: dict[str, AgentSpec]
    agent: str
    tools: dict[str, Tool]
    workspace: Path
    model: object = None
    models: dict[str, ModelChoice] = field(default_factory=dict)
    device: str | None = None
    engine: str = TRANSFORMERS
    history: list[dict[str, str]] = field(default_factory=list)
    running: bool = True
    trace: str = "on"
    stream: bool = True
    last: AgentResult | None = None
    last_spec: AgentSpec | None = None
    streamed: bool = False

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


# A command and its plural differ by one letter and mean different things, so
# `/agents python` used to list the agents and quietly drop the argument. Both
# spellings now work either way: with a name they switch, without one they list.
SYNONYMS = {"agents": "agent", "models": "model", "tool": "tools", "route": "routes"}


def run_command(session: Session, line: str) -> str:
    name, _, argument = line[1:].partition(" ")
    argument = argument.strip()
    name = SYNONYMS.get(name, name) if argument or name in {"models", "tool", "route"} else name

    if name in {"quit", "exit", "q"}:
        session.running = False
        return ""
    if name == "help":
        return HELP
    if name == "agents" or (name == "agent" and not argument):
        rows = [f"{'*' if session.agent == AUTO else ' '} {AUTO:9s} route by the prompt"]
        rows += [
            f"{'*' if key == session.agent else ' '} {key:9s} {list(spec.tools) or '-'}"
            for key, spec in session.agents.items()
        ]
        rows.append("")
        rows.append("switch with /agent NAME")
        return "\n".join(rows)
    if name == "agent":
        if argument != AUTO and argument not in session.agents:
            return f"unknown agent {argument!r}; try {[AUTO, *sorted(session.agents)]}"
        session.agent = argument
        session.history.clear()
        return f"agent: {argument}"
    if name == "stream":
        if argument:
            if argument not in {"on", "off"}:
                return "usage: /stream on|off"
            session.stream = argument == "on"
        state = "on" if session.stream else "off"
        return f"stream: {state}" + ("" if session.stream else " (answers are formatted instead)")
    if name == "decoding":
        return set_decoding(session, argument)
    if name in {"inference", "engine"}:
        return switch_engine(session, argument)
    if name == "model":
        return switch_model(session, argument)
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
        others = " | ".join(level for level in LEVELS if level != session.trace)
        return f"trace: {session.trace} · change with /trace {others}"
    if name == "last":
        if session.last is None:
            return "nothing to replay yet"
        lines = render_turn(session.last, level="full", color=use_color())
        return "\n".join([*lines, format_answer(session.last.answer, use_color())])
    if name == "new":
        session.history.clear()
        return "conversation cleared"
    return f"unknown command {line.split()[0]!r}; /help lists them"


TUNABLE = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_new_tokens")


def set_decoding(session: Session, argument: str) -> str:
    """Show the current agent's decoding settings, or override them.

    Overrides live on the agent for the rest of the session, so `/decoding
    temperature=0` makes the current agent deterministic without editing a file.
    """
    spec = session.spec
    if not argument:
        current = spec.decoding
        settings = "  ".join(f"{name}={getattr(current, name)}" for name in TUNABLE)
        return f"{spec.name}: {settings}\n\nset with /decoding temperature=0.7 top_p=0.95"

    changes: dict[str, float | int] = {}
    for pair in argument.split():
        if "=" not in pair:
            return f"expected name=value, got {pair!r}"
        name, _, raw = pair.partition("=")
        if name not in TUNABLE:
            return f"unknown setting {name!r}; try {list(TUNABLE)}"
        try:
            changes[name] = int(raw) if name in {"top_k", "max_new_tokens"} else float(raw)
        except ValueError:
            return f"{name} needs a number, got {raw!r}"

    try:
        session.agents[spec.name] = replace(spec, **changes)
    except ValueError as exc:
        return str(exc)
    return set_decoding(session, "")


def register(session: Session, argument: str) -> str:
    """Add a Hugging Face model to the list, then load it.

    The id is checked against the hub first: a typo caught here costs a second,
    and caught later costs a failed download.
    """
    model_id, _, alias = argument.partition(" ")
    if not model_id:
        return "usage: /model add ORG/NAME [alias]"
    try:
        note = describe_hub_model(model_id)
        choice = register_model(model_id, alias=alias.strip() or None, note=note)
    except Exception as exc:
        return f"{exc}"

    session.models = load_models()
    return f"registered {choice.alias} -> {choice.id}\n{switch_model(session, choice.alias)}"


def forget(session: Session, alias: str) -> str:
    if not alias:
        return "usage: /model remove ALIAS"
    try:
        removed = unregister_model(alias)
    except Exception as exc:
        return f"{exc}"
    session.models = load_models()
    return f"removed {alias} ({removed}); the loaded model is unchanged"


def switch_model(session: Session, argument: str) -> str:
    current = getattr(session.model, "model_id", "")
    if not argument:
        rows = []
        width = max(len(name) for name in [*session.models, "(unlisted)"])
        for alias, choice in session.models.items():
            mark = "*" if choice.id == current else " "
            rows.append(f"{mark} {alias:{width}s}  {choice.note}")
        if current not in {choice.id for choice in session.models.values()}:
            rows.append(f"* {'(unlisted)':{width}s}  {current}")
        rows.append("")
        rows.append("load one with /model ALIAS, or /model any/huggingface-id")
        rows.append("register a new one with /model add ORG/NAME")
        return "\n".join(rows)

    verb, _, rest = argument.partition(" ")
    if verb == "add":
        return register(session, rest.strip())
    if verb in {"remove", "forget"}:
        return forget(session, rest.strip())

    choice = resolve_model(argument, session.models)
    if choice.id == current:
        return f"already on {choice.id}"
    if choice.alias not in session.models:
        # Registered models were checked when they were added; an id typed
        # straight in has not been.
        try:
            check_size(choice.id)
        except ValueError as exc:
            return str(exc)

    note = " (running custom code from its repository)" if choice.trust_remote_code else ""
    session.model.unload()
    with working(f"Loading {choice.id}{note}", stream=sys.stdout):
        session.model = load_model(choice, session.device, session.engine)
    session.history.clear()
    return f"model: {choice.id} · tool calls: {session.model.call_format.name}"


def switch_engine(session: Session, argument: str) -> str:
    """Move the current model onto a different engine.

    The served engines need their server already running with this model; the
    error says so rather than leaving the session without a model.
    """
    if not argument:
        others = " ".join(name for name in ENGINES if name != session.engine)
        return f"inference: {session.engine} · switch with /inference {others}"
    if argument not in ENGINES:
        return f"unknown engine {argument!r}; try {list(ENGINES)}"
    if argument == session.engine:
        return f"already on {argument}"

    choice = resolve_model(getattr(session.model, "model_id", ""), session.models)
    previous, session.engine = session.engine, argument
    try:
        with working(f"Switching to {argument}", stream=sys.stdout):
            replacement = load_model(choice, session.device, argument)
    except EngineUnavailable as exc:
        session.engine = previous
        return f"{exc}"
    session.model.unload()
    session.model = replacement
    session.history.clear()
    return f"inference: {argument} · {session.model.device}"


def answer(session: Session, prompt: str) -> AgentResult:
    model = session.model
    color = use_color()
    spec, rule = session.resolve(prompt)

    if session.agent == AUTO and session.trace != "off":
        print(render_route(spec.name, rule, color))

    def show(step: Step) -> None:
        for line in render_step(step, session.trace, color):
            print(line)

    gate = None
    if session.stream:
        gate = StreamGate(
            model.call_format.marker,
            lambda text: print(text, end="", flush=True),
            thinking=getattr(model, "thinks_first", False),
        )

    result = run_agent(
        model,
        spec,
        prompt,
        session.tools,
        history=session.history,
        on_step=show,
        on_token=gate.feed if gate else None,
        agents=session.agents,
    )
    if gate:
        gate.close()
        if gate.emitted:
            print()
    session.streamed = bool(gate and gate.emitted)
    session.remember(prompt, result.answer)
    session.last = result
    session.last_spec = spec
    return result


def start_terminal(session: Session) -> int:
    """Read prompts until end of input, keeping the model loaded between them."""
    # Loading the model logs at INFO; nothing below that belongs in a session.
    logging.getLogger().setLevel(logging.WARNING)
    if HISTORY_FILE.exists():
        readline.read_history_file(HISTORY_FILE)

    print(BANNER.format(model=session.model.model_id.split("/")[-1], agent=session.agent))
    while session.running:
        try:
            line = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            try:
                message = run_command(session, line)
            except Exception as exc:
                # A command failing is not a reason to lose the loaded model
                # and the conversation; say what went wrong and carry on.
                logger.debug("command failed", exc_info=True)
                message = f"{line.split()[0]} failed: {type(exc).__name__}: {exc}"
            if message:
                print(message)
            continue

        try:
            result = answer(session, line)
        except KeyboardInterrupt:
            print("\ninterrupted")
            continue
        spec = session.last_spec or session.spec
        # A streamed answer is already on screen; printing it again would double it.
        if not session.streamed:
            print(format_answer(result.answer, use_color()))
        if session.trace != "off":
            print(render_summary(result, len(spec.tools), color=use_color()))
            if session.agent == AUTO and not spec.tools:
                print(NO_TOOL_HINT.format(agent=spec.name))
            elif session.agent != AUTO:
                # Pinning is easy to forget: `write python code for X` runs it
                # rather than writing it if the session is still on python.
                elsewhere = choose_agent(line, set(session.agents))
                if elsewhere.matched and elsewhere.agent != session.agent:
                    print(PINNED_HINT.format(agent=session.agent, other=elsewhere.agent))

    readline.write_history_file(HISTORY_FILE)
    return 0
