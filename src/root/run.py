from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .agent import run_agent
from .backend import LocalModel
from .config import DEFAULT_MODEL, load_agents
from .terminal import Session, start_terminal
from .tools import build_tools
from .trace import LEVELS, render_step, render_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="root", description=__doc__)
    parser.add_argument("prompt", nargs="*", help="prompt text; read from stdin when omitted")
    parser.add_argument("--agent", default="chat", help="agent name from the config")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or path")
    parser.add_argument("--config", type=Path, default=Path("configs/agents.yaml"))
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--device", help="cuda, mps or cpu; auto-detected when omitted")
    parser.add_argument("--max-steps", type=int, help="override the agent's step budget")
    parser.add_argument("--output-dir", type=Path, help="write result.json here")
    parser.add_argument("--list", action="store_true", help="list configured agents and exit")
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

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1
    agents = load_agents(args.config)

    if args.list:
        for name, spec in agents.items():
            print(f"{name:12s} tools={list(spec.tools) or '-'} max_steps={spec.max_steps}")
        return 0

    if args.agent not in agents:
        print(f"Unknown agent {args.agent!r}; have {sorted(agents)}", file=sys.stderr)
        return 1
    spec = agents[args.agent]
    if args.max_steps:
        spec = replace(spec, max_steps=args.max_steps)

    tools = build_tools(args.workspace)
    unknown = set(spec.tools) - set(tools)
    if unknown:
        print(f"Agent {spec.name} requests unknown tools: {sorted(unknown)}", file=sys.stderr)
        return 1

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    model = LocalModel.load(args.model, device=args.device)

    # No prompt and a terminal attached means the user wants the interactive
    # session, where the model is loaded once and reused.
    if not prompt:
        session = Session(
            agents=agents,
            agent=spec.name,
            tools=tools,
            workspace=args.workspace.resolve(),
            trace=args.trace,
        )
        return start_terminal(model, session)

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
            "model": args.model,
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
