"""Expose root's agent loop as a Harbor agent.

Harbor drives the benchmark - it builds the task container, runs the agent
against the instruction, then grades the container with the task's own verifier.
root plugs in as an agent rather than reimplementing any of that:

    harbor run -d terminal-bench-science/terminal-bench-science@latest \\
      --agent root.harbor:RootAgent \\
      --model lfm2-350m \\
      --env singularity

`--ak key=value` reaches this class as keyword arguments: `agent` picks which
root agent to run (default `terminal`), `engine` and `engine_url` choose what
generates the tokens, and `max_steps` overrides the step budget. Serving a large
model with `--ak engine=vllm` is the interesting configuration; the sub-1B
models root ships cannot do this benchmark's tasks at all.

Requires the `harbor` extra: `uv sync --extra harbor`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from . import __version__
from .config import AgentSpec, config_path, load_agents, load_models, resolve_model
from .engines import AUTO, load_engine
from .sandbox import MAX_OUTPUT_CHARS, Command, solve, transcript, usage

logger = logging.getLogger(__name__)

DEFAULT_AGENT = "terminal"
# One command should not be able to hang the whole trial; Harbor's own agent
# timeout is hours, and a task that waits forever on stdin would spend them.
COMMAND_TIMEOUT_SEC = 600


class RootAgent(BaseAgent):
    """root's loop, with its shell tool pointed at the task container.

    The model stays on the host, which is what makes this workable: the task
    environments are CPU-only, and the weights want the GPU the host has.
    """

    @staticmethod
    @override
    def name() -> str:
        return "root"

    @override
    def version(self) -> str:
        return __version__

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *args,
        agent: str = DEFAULT_AGENT,
        engine: str = AUTO,
        engine_url: str | None = None,
        max_steps: int | None = None,
        max_output_chars: int = MAX_OUTPUT_CHARS,
        **kwargs,
    ):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self.agent_name = agent
        self.engine = engine
        self.engine_url = engine_url
        self.max_steps = max_steps
        self.max_output_chars = max_output_chars
        self.model = None

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """Load the model on the host; nothing is installed in the container.

        Loading here rather than in `run` keeps model download and weight
        loading out of the task's own agent timeout.
        """
        choice = resolve_model(self.model_name or "", load_models())
        self.logger.info("loading %s on the %s engine", choice.id, self.engine)
        self.model = await asyncio.to_thread(
            load_engine,
            choice.id,
            self.engine,
            base_url=self.engine_url,
            trust_remote_code=choice.trust_remote_code,
        )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        spec = self._spec()
        loop = asyncio.get_running_loop()

        def run_command(command: str) -> Command:
            """Bridge the loop's synchronous tool call onto the event loop.

            `run_agent` is synchronous and runs on a worker thread, while
            `environment.exec` is a coroutine belonging to this loop, so the
            call has to be handed back rather than awaited in place.
            """
            future = asyncio.run_coroutine_threadsafe(
                environment.exec(command, timeout_sec=COMMAND_TIMEOUT_SEC), loop
            )
            result = future.result()
            return Command(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                return_code=result.return_code,
            )

        attempt = await asyncio.to_thread(
            solve, self.model, spec, instruction, run_command, self.max_output_chars
        )
        result = attempt.result

        tokens = usage(result)
        context.n_input_tokens = tokens["n_input_tokens"]
        context.n_output_tokens = tokens["n_output_tokens"]
        context.metadata = {
            "root_agent": spec.name,
            "model": getattr(self.model, "model_id", self.model_name),
            "engine": getattr(self.model, "engine", self.engine),
            "commands": len(attempt.commands),
            "steps": len(result.steps),
            "hit_step_limit": result.hit_step_limit,
            "answer": result.answer,
        }
        (self.logs_dir / "root-answer.txt").write_text(result.answer, encoding="utf-8")
        (self.logs_dir / "root-transcript.txt").write_text(transcript(attempt), encoding="utf-8")

    def _spec(self) -> AgentSpec:
        agents = load_agents(config_path("agents.yaml"))
        if self.agent_name not in agents:
            raise ValueError(
                f"unknown root agent {self.agent_name!r}; configured: {sorted(agents)}"
            )
        spec = agents[self.agent_name]
        if self.max_steps is not None:
            spec = replace(spec, max_steps=self.max_steps)
        return spec
