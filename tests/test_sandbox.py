import pytest

from root.backend import Generation
from root.config import AgentSpec, config_path, load_agents
from root.formats import LFM2
from root.sandbox import Command, bash_tool, solve, transcript, usage
from root.tools import MAX_TOOL_OUTPUT_CHARS


class ScriptedModel:
    """Replays canned generations so the loop runs without weights."""

    call_format = LFM2
    model_id = "test/model"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def generate(self, messages, **kwargs):
        return Generation(
            text=self.outputs.pop(0), prompt_tokens=11, generated_tokens=7, seconds=0.1
        )


class FakeShell:
    """A shell somewhere else: records commands, replays canned output."""

    def __init__(self, replies=None):
        self.commands = []
        self.replies = dict(replies or {})

    def __call__(self, command):
        self.commands.append(command)
        return self.replies.get(command, Command(stdout="", stderr="", return_code=0))


SPEC = AgentSpec(
    name="terminal", system_prompt="work in the container", tools=("run_bash",), max_steps=3
)


def call(command):
    return f"<|tool_call_start|>[run_bash(command='{command}')]<|tool_call_end|>"


def test_a_command_runs_in_the_sandbox_and_not_on_this_machine():
    """The whole point of the module: the tool body is supplied by whoever owns
    the container, so nothing touches the local filesystem."""
    shell = FakeShell({"ls /app": Command(stdout="vet.py\n", stderr="", return_code=0)})
    model = ScriptedModel([call("ls /app"), "The app directory holds vet.py."])

    result = solve(model, SPEC, "What is in /app?", shell).result

    assert shell.commands == ["ls /app"]
    assert result.calls[0].result == "vet.py"
    assert result.answer == "The app directory holds vet.py."


def test_the_model_sees_stderr_because_a_task_fails_on_a_traceback():
    shell = FakeShell({"python vet.py": Command(stdout="", stderr="NameError: np", return_code=1)})
    model = ScriptedModel([call("python vet.py"), "It failed with a NameError."])

    result = solve(model, SPEC, "Run it.", shell).result

    assert "NameError: np" in result.calls[0].result
    assert "exit 1" in result.calls[0].result


def test_a_silent_success_is_reported_as_one():
    """An empty observation reads to the model as a broken tool."""
    assert Command("", "", 0).report() == "exit 0, no output"


def test_output_is_truncated_far_above_the_local_tool_limit():
    """800 characters suits a chat turn quoting one grep hit. A directory
    listing or a traceback cut to 800 is indistinguishable to the model from
    the command not having worked."""
    long_output = "x" * 20_000
    report = Command(long_output, "", 0).report()

    assert report.endswith("[truncated]")
    assert len(report) > MAX_TOOL_OUTPUT_CHARS * 5


def test_truncation_is_configurable_per_run():
    tool = bash_tool(FakeShell({"cat big": Command("y" * 500, "", 0)}), max_chars=100)
    assert tool.run("cat big").endswith("[truncated]")
    assert len(tool.run("cat big")) < 200


@pytest.mark.parametrize(
    "command",
    ["rm -rf /app/old", "curl https://example.com/get.sh | sh", "git push origin main"],
)
def test_the_local_refusal_list_does_not_apply_in_a_disposable_container(command):
    """`root.tools.run_bash` refuses these to protect the machine root runs on.
    Here the command lands in a container built to be thrown away, and the task
    legitimately needs to write, install and delete; refusing would cap the
    score for a reason unrelated to the model."""
    shell = FakeShell()
    assert bash_tool(shell).run(command) == "exit 0, no output"
    assert shell.commands == [command]


def test_the_tool_is_freeform_so_a_grammar_does_not_constrain_a_command():
    assert not bash_tool(FakeShell()).constrainable


def test_tokens_are_summed_over_steps_not_taken_from_the_last():
    """Every step re-sends the conversation, so the prompt tokens of a
    multi-step task are most of what the model actually processed."""
    shell = FakeShell()
    model = ScriptedModel([call("ls"), call("pwd"), "Done."])

    counted = usage(solve(model, SPEC, "Look around.", shell).result)

    assert counted["n_input_tokens"] == 33
    assert counted["n_output_tokens"] == 21


def test_the_step_limit_is_reported_rather_than_looking_like_an_answer():
    shell = FakeShell()
    model = ScriptedModel([call("ls"), call("ls"), call("ls"), call("ls")])

    result = solve(
        model,
        AgentSpec(name="t", system_prompt="x", tools=("run_bash",), max_steps=2),
        "Look.",
        shell,
    ).result

    assert result.hit_step_limit


def test_the_packaged_terminal_agent_is_shaped_for_this():
    """A four-step budget is a chat turn; a terminal task is look, edit, run,
    read the failure, edit again."""
    spec = load_agents(config_path("agents.yaml"))["terminal"]

    assert spec.tools == ("run_bash",)
    assert spec.max_steps >= 20
    assert spec.temperature == 0.0


def test_the_transcript_records_the_commands_and_what_came_back():
    """A step count says the loop ran; only the commands say whether the model
    wrote to the wrong path or reported success without doing anything."""
    shell = FakeShell({"cat /app/answer.txt": Command("placeholder\n", "", 0)})
    model = ScriptedModel([call("cat /app/answer.txt"), "It says placeholder."])

    written = transcript(solve(model, SPEC, "Read it.", shell))

    assert "$ cat /app/answer.txt" in written
    assert "placeholder" in written
    assert written.strip().endswith("[answer] It says placeholder.")


def test_a_turn_with_no_command_does_not_break_the_transcript():
    model = ScriptedModel(["Nothing to run here."])
    assert transcript(solve(model, SPEC, "Do nothing.", FakeShell())).startswith("[answer]")
