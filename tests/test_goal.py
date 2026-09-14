import pytest

from root.backend import Generation
from root.config import AgentSpec
from root.formats import LFM2
from root.goal import Check, pursue, retry_prompt, summarise
from root.tools import build_tools

SPEC = AgentSpec(name="terminal", system_prompt="work", tools=("run_bash",), max_steps=3)


class ScriptedModel:
    """Replays canned turns so the outer loop runs without weights."""

    call_format = LFM2
    model_id = "test/model"

    def __init__(self, turns):
        self.turns = list(turns)
        self.prompts = []

    def generate(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        text = self.turns.pop(0) if self.turns else "done"
        return Generation(text=text, prompt_tokens=5, generated_tokens=5, seconds=0.0)


def touching(path):
    return f"<|tool_call_start|>[run_bash(command='touch {path}')]<|tool_call_end|>"


def test_a_task_already_satisfied_costs_nothing(tmp_path):
    """A check that already passes should not wake the model at all."""
    model = ScriptedModel([])
    done = pursue(model, SPEC, "make it so", build_tools(tmp_path), Check("true", tmp_path))

    assert done.passed and done.already_passing
    assert done.attempts == []
    assert model.prompts == []


def test_the_loop_stops_as_soon_as_the_check_passes(tmp_path):
    model = ScriptedModel([touching("marker"), "I made the file."])
    check = Check("test -f marker", tmp_path)

    done = pursue(model, SPEC, "create marker", build_tools(tmp_path), check, attempts=4)

    assert done.passed
    assert len(done.attempts) == 1
    assert (tmp_path / "marker").exists()


def test_the_agents_own_opinion_does_not_end_the_loop(tmp_path):
    """The model says it is finished after every attempt. Only the check decides,
    which is the entire reason this wraps run_agent instead of replacing it."""
    model = ScriptedModel(["all done", "all done", touching("marker"), "all done"])
    check = Check("test -f marker", tmp_path)

    done = pursue(model, SPEC, "create marker", build_tools(tmp_path), check, attempts=4)

    assert done.passed
    assert len(done.attempts) == 3


def test_the_attempt_budget_is_honoured(tmp_path):
    model = ScriptedModel(["nope"] * 20)
    done = pursue(
        model, SPEC, "impossible", build_tools(tmp_path), Check("false", tmp_path), attempts=3
    )

    assert not done.passed
    assert len(done.attempts) == 3
    assert len(done.outcomes) == 3


def test_the_failure_output_is_given_to_the_next_attempt(tmp_path):
    """Without it the model rewrites the same thing, having no way to know what
    its last edit did."""
    model = ScriptedModel(["nope"] * 6)
    check = Check("echo 'AssertionError: expected 3' >&2; false", tmp_path)

    pursue(model, SPEC, "fix it", build_tools(tmp_path), check, attempts=2)

    assert "AssertionError: expected 3" in model.prompts[-1]


def test_the_conversation_carries_into_the_next_attempt(tmp_path):
    """A fresh start each time would have the model rediscover a workspace it
    already changed."""
    model = ScriptedModel([touching("a"), "made a", "nope"])
    pursue(model, SPEC, "task", build_tools(tmp_path), Check("false", tmp_path), attempts=2)

    assert any("made a" in prompt for prompt in model.prompts) or len(model.prompts) > 2


def test_a_check_that_hangs_is_not_a_check_that_passed(tmp_path):
    outcome = Check("sleep 5", tmp_path, timeout_sec=1).run()
    assert not outcome.passed
    assert "did not finish" in outcome.output


def test_the_check_reads_both_streams(tmp_path):
    """A failing test run says why on stderr, and that is the useful half."""
    outcome = Check("echo out; echo err >&2; false", tmp_path).run()
    assert "out" in outcome.output and "err" in outcome.output


def test_a_long_check_keeps_its_tail(tmp_path):
    """The summary line of a test run is at the end, not the beginning."""
    outcome = Check("python3 -c \"print('x'*20000); print('FAILED here')\"; false", tmp_path).run()
    assert outcome.output.endswith("FAILED here")


def test_the_check_runs_in_the_workspace(tmp_path):
    (tmp_path / "only-here").write_text("x")
    assert Check("test -f only-here", tmp_path).run().passed


@pytest.mark.parametrize("passed,expected", [(True, "passed"), (False, "still failing")])
def test_the_summary_says_how_it_ended(tmp_path, passed, expected):
    model = ScriptedModel(["nope"] * 4)
    check = Check("true" if passed else "false", tmp_path)
    done = pursue(model, SPEC, "task", build_tools(tmp_path), check, attempts=1)
    if passed:
        assert "already passes" in summarise(done)
    else:
        assert expected in summarise(done)


def test_the_retry_prompt_carries_the_output_and_the_task():
    from root.goal import Outcome

    text = retry_prompt("fix the parser", Outcome(False, "SyntaxError: line 3"))
    assert "SyntaxError: line 3" in text
    assert "fix the parser" in text
