from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from .agent import run_agent
from .backend import LocalModel
from .config import (
    PACKAGED_CONFIGS,
    config_path,
    load_agents,
    load_default_model,
    load_models,
    resolve_model,
)
from .evals import SCRATCH, accuracy, by_agent, load_cases, score_case
from .tools import build_tools

logger = logging.getLogger(__name__)


def case_workspace(name: str | None, evals: Path, fallback: Path) -> Path:
    """Where a case's workspace lives.

    Relative to the eval file that named it, falling back to the packaged
    fixture. A local configs/ that holds the YAML but not the fixture would
    otherwise point every file case at a directory that does not exist, and
    they would fail as though the model had got it wrong.
    """
    if not name:
        return fallback
    beside = evals.parent / name
    if beside.is_dir():
        return beside
    packaged = PACKAGED_CONFIGS / name
    if packaged.is_dir():
        logger.info("case workspace %s not beside %s; using the packaged one", name, evals)
        return packaged
    return beside


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="root-eval",
        description="Score agents on tool choice and answer content.",
    )
    parser.add_argument("--evals", type=Path, help="eval cases; defaults to the packaged set")
    parser.add_argument("--config", type=Path, help="agent config; defaults to the packaged one")
    parser.add_argument("--model", default=load_default_model())
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--device")
    parser.add_argument(
        "--no-grammar", action="store_true", help="do not constrain forced tool calls"
    )
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
    if args.no_grammar:
        os.environ["ROOT_NO_GRAMMAR"] = "1"

    agents = load_agents(config_path("agents.yaml", args.config))
    evals = config_path("evals.yaml", args.evals)
    cases = load_cases(evals)
    if args.agent:
        cases = [case for case in cases if case.agent == args.agent]
    unknown = {case.agent for case in cases} - set(agents)
    if unknown:
        print(f"Eval cases reference unknown agents: {sorted(unknown)}", file=sys.stderr)
        return 1
    if not cases:
        print("No cases to run", file=sys.stderr)
        return 1

    choice = resolve_model(args.model, load_models())
    model = LocalModel.load(
        choice.id, device=args.device, trust_remote_code=choice.trust_remote_code
    )
    # A case may point at a fixture directory so its answer does not shift when
    # the repository it would otherwise search changes.
    # A case workspace is relative to the eval file, so a packaged fixture is
    # found wherever the command runs from. `tmp` means a fresh empty directory,
    # which is what a case that writes files needs.
    fixed = {
        name: build_tools(case_workspace(name, evals, args.workspace))
        for name in {case.workspace for case in cases}
        if name != SCRATCH
    }

    scores = []
    with ExitStack() as scratch_dirs:
        for case in cases:
            if case.workspace == SCRATCH:
                tools = build_tools(Path(scratch_dirs.enter_context(tempfile.TemporaryDirectory())))
            else:
                tools = fixed[case.workspace]
            result = run_agent(model, agents[case.agent], case.prompt, tools)
            score = score_case(case, result)
            scores.append(score)
            mark = "PASS" if score.passed else "FAIL"
            print(f"{mark} [{case.agent}] {case.prompt[:64]}")
            print(f"     called={score.tools_called} answer={' '.join(score.answer.split())[:90]}")
            for reason in score.failures():
                print(f"     -> {reason}")

    rates = accuracy(scores)
    passed = sum(s.passed for s in scores)

    grouped = by_agent(scores)
    if len(grouped) > 1:
        print()
        for agent, agent_scores in grouped.items():
            agent_rates = accuracy(agent_scores)
            hits = sum(s.passed for s in agent_scores)
            print(
                f"  {agent:9s} {hits:3d}/{len(agent_scores):<3d} "
                f"tool {agent_rates['tool']:>4.0%}  answer {agent_rates['answer']:>4.0%}"
            )
    print(
        f"\n{passed}/{len(scores)} cases passed "
        f"(tool {rates['tool']:.0%}, answer {rates['answer']:.0%})"
    )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model.model_id,
            "device": model.device,
            "accuracy": rates,
            "by_agent": {
                agent: accuracy(agent_scores) | {"cases": len(agent_scores)}
                for agent, agent_scores in grouped.items()
            },
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
