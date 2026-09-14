"""The Harbor glue. Needs the `harbor` extra, which is 89 packages, so these
skip on a default `uv sync --extra dev`; the logic they cover lives in
`root.sandbox` and is tested there unconditionally."""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("harbor.agents.base", reason="install the harbor extra to exercise the adapter")

from harbor.environments.base import ExecResult  # noqa: E402
from harbor.models.agent.context import AgentContext  # noqa: E402

from root.backend import Generation  # noqa: E402
from root.formats import LFM2  # noqa: E402
from root.harbor import RootAgent  # noqa: E402


class ScriptedModel:
    call_format = LFM2
    model_id = "test/model"
    engine = "transformers"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def generate(self, messages, **kwargs):
        return Generation(
            text=self.outputs.pop(0), prompt_tokens=11, generated_tokens=7, seconds=0.1
        )


class FakeEnvironment:
    """Stands in for a task container: records commands, replays output."""

    default_user = None

    def __init__(self, replies=None):
        self.commands = []
        self.replies = dict(replies or {})

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        stdout, code = self.replies.get(command, ("", 0))
        return ExecResult(stdout=stdout, stderr="", return_code=code)


def call(command):
    return f"<|tool_call_start|>[run_bash(command='{command}')]<|tool_call_end|>"


def build(tmp_path, outputs, **kwargs):
    agent = RootAgent(logs_dir=Path(tmp_path), model_name="lfm2-350m", **kwargs)
    agent.model = ScriptedModel(outputs)
    return agent


def test_the_agent_drives_commands_into_the_harbor_environment(tmp_path):
    environment = FakeEnvironment({"ls /app": ("vet.py\n", 0)})
    agent = build(tmp_path, [call("ls /app"), "It holds vet.py."])
    context = AgentContext()

    asyncio.run(agent.run("What is in /app?", environment, context))

    assert environment.commands == ["ls /app"]
    assert context.metadata["commands"] == 1
    assert context.metadata["answer"] == "It holds vet.py."


def test_the_context_carries_the_token_counts_harbor_reports(tmp_path):
    agent = build(tmp_path, [call("pwd"), "Done."])
    context = AgentContext()

    asyncio.run(agent.run("Where am I?", FakeEnvironment(), context))

    assert context.n_input_tokens == 22
    assert context.n_output_tokens == 14
    assert not context.is_empty()


def test_the_answer_is_written_beside_the_trial_logs(tmp_path):
    agent = build(tmp_path, ["Nothing to do."])
    asyncio.run(agent.run("Do nothing.", FakeEnvironment(), AgentContext()))
    assert (tmp_path / "root-answer.txt").read_text() == "Nothing to do."


def test_the_step_budget_can_be_overridden_from_the_command_line(tmp_path):
    """`--ak max_steps=8` has to reach the spec, or a long task is cut off at
    whatever agents.yaml happens to say."""
    agent = build(tmp_path, ["done"], max_steps=8)
    assert agent._spec().max_steps == 8


def test_an_unknown_root_agent_fails_loudly(tmp_path):
    agent = build(tmp_path, ["done"], agent="not-an-agent")
    with pytest.raises(ValueError, match="unknown root agent"):
        agent._spec()


def test_the_adapter_names_itself_for_harbor(tmp_path):
    agent = build(tmp_path, [])
    assert agent.name() == "root"
    assert agent.version()
