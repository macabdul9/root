from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import __version__
from .agent import run_agent
from .backend import LocalModel
from .config import (
    config_path,
    copy_defaults,
    load_agents,
    load_default_model,
    load_models,
    resolve_model,
)
from .route import AUTO, Route, choose_agent
from .terminal import Session, start_terminal
from .tools import build_tools
from .trace import LEVELS, render_route, render_step, render_summary


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
    parser.add_argument("--max-steps", type=int, help="override the agent's step budget")
    parser.add_argument("--output-dir", type=Path, help="write result.json here")
    parser.add_argument("--version", action="version", version=f"root {__version__}")
    parser.add_argument("--list", action="store_true", help="list configured agents and exit")
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
    unknown = {name for spec in agents.values() for name in spec.tools} - set(tools)
    if unknown:
        print(f"Agents request unknown tools: {sorted(unknown)}", file=sys.stderr)
        return 1

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    models = load_models()
    choice = resolve_model(args.model, models)
    model = LocalModel.load(
        choice.id, device=args.device, trust_remote_code=choice.trust_remote_code
    )

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
            trace=args.trace,
        )
        return start_terminal(session)

    route = choose_agent(prompt, set(agents)) if args.agent == AUTO else Route(args.agent, None)
    spec = agents[route.agent]
    if args.max_steps:
        spec = replace(spec, max_steps=args.max_steps)
    if args.agent == AUTO and args.trace != "off":
        print(render_route(spec.name, route.rule), file=sys.stderr)

    def show(step):
        for line in render_step(step, args.trace):
            print(line, file=sys.stderr)

    result = run_agent(model, spec, prompt, tools, on_step=show)
    print(result.answer, flush=True)
    if args.trace != "off":
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
