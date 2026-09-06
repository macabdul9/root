from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .agent import run_agent
from .backend import LocalModel
from .config import DEFAULT_MODEL, load_agents
from .evals import accuracy, load_cases, score_case
from .tools import build_tools


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="root-eval",
        description="Score agents on tool choice and answer content.",
    )
    parser.add_argument("--evals", type=Path, default=Path("configs/evals.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/agents.yaml"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--device")
    parser.add_argument("--agent", help="only run cases for this agent")
    parser.add_argument("--output-dir", type=Path, help="write evals.json here")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="exit non-zero when overall accuracy falls below this",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    agents = load_agents(args.config)
    cases = load_cases(args.evals)
    if args.agent:
        cases = [case for case in cases if case.agent == args.agent]
    unknown = {case.agent for case in cases} - set(agents)
    if unknown:
        print(f"Eval cases reference unknown agents: {sorted(unknown)}", file=sys.stderr)
        return 1
    if not cases:
        print("No cases to run", file=sys.stderr)
        return 1

    model = LocalModel.load(args.model, device=args.device)
    # A case may point at a fixture directory so its answer does not shift when
    # the repository it would otherwise search changes.
    workspaces = {case.workspace for case in cases}
    tools = {name: build_tools(Path(name or args.workspace)) for name in workspaces}

    scores = []
    for case in cases:
        result = run_agent(model, agents[case.agent], case.prompt, tools[case.workspace])
        score = score_case(case, result)
        scores.append(score)
        mark = "PASS" if score.passed else "FAIL"
        print(f"{mark} [{case.agent}] {case.prompt[:64]}")
        print(f"     called={score.tools_called} answer={' '.join(score.answer.split())[:90]}")
        for reason in score.failures():
            print(f"     -> {reason}")

    rates = accuracy(scores)
    passed = sum(s.passed for s in scores)
    print(
        f"\n{passed}/{len(scores)} cases passed "
        f"(tool {rates['tool']:.0%}, answer {rates['answer']:.0%})"
    )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "device": model.device,
            "accuracy": rates,
            "cases": [
                {
                    **asdict(score.case),
                    "tools_called": score.tools_called,
                    "answer": score.answer,
                    "passed": score.passed,
                    "failures": score.failures(),
                }
                for score in scores
            ],
        }
        (args.output_dir / "evals.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return 0 if rates["overall"] >= args.min_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())
