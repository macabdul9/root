from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import __version__
from .agent import run_agent
from .config import (
    config_path,
    copy_defaults,
    load_agents,
    load_default_model,
    load_models,
    resolve_model,
)
from .engines import AUTO as AUTO_ENGINE
from .engines import ENGINES, EngineUnavailable, load_engine
from .mcp import MCPTools, declared_tools, describe, load_servers
from .progress import working
from .route import AUTO, Route, choose_agent, fallback_agent
from .terminal import Session, start_terminal
from .tools import build_tools
from .trace import LEVELS, StreamGate, render_route, render_step, render_summary

DECODING_FLAGS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "max_new_tokens",
        "seed",
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="root", description=__doc__)
    parser.add_argument("prompt", nargs="*", help="prompt text; read from stdin when omitted")
    parser.add_argument(
        "--agent",
        default="auto",
        help="agent name from the config, or 'auto' to route by the prompt",
    )
    parser.add_argument(
        "--model",
        default=load_default_model(),
        help="Hugging Face model id, local path, or an alias from models.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="agent config; defaults to ./configs/agents.yaml, else the packaged one",
    )
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--device", help="cuda, mps or cpu; auto-detected when omitted")
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default=AUTO_ENGINE,
        help="what generates the tokens; auto picks the fastest one already serving",
    )
    parser.add_argument("--engine-url", help="base URL of the vllm, sglang or ollama server")
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="do not constrain forced tool calls; faster, and the parser repairs instead",
    )
    parser.add_argument("--max-steps", type=int, help="override the agent's step budget")
    parser.add_argument("--temperature", type=float, help="0 for greedy decoding")
    parser.add_argument("--top-p", type=float, help="nucleus sampling mass")
    parser.add_argument("--top-k", type=int, help="0 disables top-k")
    parser.add_argument("--min-p", type=float, help="0 disables min-p")
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, help="make sampling reproducible")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="print the answer as it is generated, instead of formatted at the end",
    )
    parser.add_argument("--output-dir", type=Path, help="write result.json here")
    parser.add_argument("--version", action="version", version=f"root {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show model loading and step logs"
    )
    parser.add_argument("--list", action="store_true", help="list configured agents and exit")
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="list MCP servers, the tools they offer, and the ones root skipped, then exit",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="copy the packaged configs into ./configs for editing, then exit",
    )
    parser.add_argument(
        "--trace",
        choices=LEVELS,
        default="on",
        help="how much of the loop to print to stderr",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    if not args.verbose:
        # transformers draws a weight-loading bar of its own.
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    if args.no_grammar:
        os.environ["ROOT_NO_GRAMMAR"] = "1"
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.init:
        written = copy_defaults()
        for path in written:
            print(f"wrote {path}")
        print("already present, left alone" if not written else "edit these and rerun root")
        return 0

    config = config_path("agents.yaml", args.config)
    if not config.is_file():
        print(f"Config not found: {config}", file=sys.stderr)
        return 1
    agents = load_agents(config)

    if args.list:
        print(f"{AUTO:12s} routes to one of the agents below, by the prompt")
        for name, spec in agents.items():
            print(f"{name:12s} tools={list(spec.tools) or '-'} max_steps={spec.max_steps}")
        return 0

    if args.agent != AUTO and args.agent not in agents:
        print(f"Unknown agent {args.agent!r}; have {[AUTO, *sorted(agents)]}", file=sys.stderr)
        return 1

    tools = build_tools(args.workspace)
    servers = load_servers()
    mcp = MCPTools(servers)
    if args.mcp:
        try:
            print(describe(servers, mcp))
        finally:
            mcp.close()
        return 0
    # Only an enabled server is touched here, and touching it is what spawns it.
    if mcp.clients:
        tools |= mcp.tools()

    wanted = {name for spec in agents.values() for name in spec.tools}
    unknown = wanted - set(tools) - declared_tools(servers)
    if unknown:
        print(f"Agents request unknown tools: {sorted(unknown)}", file=sys.stderr)
        return 1

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    models = load_models()
    choice = resolve_model(args.model, models)
    try:
        with working(f"Loading {choice.id}"):
            model = load_engine(
                choice.id,
                args.engine,
                device=args.device,
                trust_remote_code=choice.trust_remote_code,
                base_url=args.engine_url,
            )
    except EngineUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:
        # A mistyped model or a repository that is gone should say so, not
        # unwind a stack of hub internals at someone typing at a prompt.
        print(f"could not load {choice.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("run with -v for the full traceback", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    # No prompt and a terminal attached means the user wants the interactive
    # session, where the model is loaded once and reused.
    if not prompt:
        session = Session(
            agents=agents,
            agent=args.agent,
            tools=tools,
            workspace=args.workspace.resolve(),
            model=model,
            models=models,
            device=args.device,
            engine=args.engine,
            trace=args.trace,
        )
        try:
            return start_terminal(session)
        finally:
            mcp.close()

    route = (
        choose_agent(prompt, set(agents), fallback_agent(agents, tools))
        if args.agent == AUTO
        else Route(args.agent, None)
    )
    spec = agents[route.agent]
    overrides = {
        name: value
        for name, value in vars(args).items()
        if value is not None and name in DECODING_FLAGS
    }
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if overrides:
        spec = replace(spec, **overrides)
    if args.agent == AUTO and args.trace != "off":
        print(render_route(spec.name, route.rule), file=sys.stderr)

    def show(step):
        for line in render_step(step, args.trace):
            print(line, file=sys.stderr)

    gate = None
    if args.stream:
        gate = StreamGate(
            model.call_format.marker,
            lambda text: print(text, end="", flush=True),
            thinking=getattr(model, "thinks_first", False),
        )

    result = run_agent(
        model,
        spec,
        prompt,
        tools,
        on_step=show,
        on_token=gate.feed if gate else None,
        agents=agents,
    )
    if gate:
        gate.close()
    if gate and gate.emitted:
        print(flush=True)
    else:
        print(result.answer, flush=True)
    if args.trace != "off" and not result.from_template:
        print(render_summary(result, len(spec.tools)), file=sys.stderr)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = asdict(result) | {
            "calls": [asdict(call) for call in result.calls],
            "model": model.model_id,
            "device": model.device,
            "seconds": result.seconds,
            "generated_tokens": result.generated_tokens,
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    mcp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
