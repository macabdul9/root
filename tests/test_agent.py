import pytest

from root.agent import AgentResult, run_agent
from root.backend import Generation
from root.config import AgentSpec, config_path, load_agents
from root.formats import LFM2
from root.tools import build_tools


class ScriptedModel:
    """Replays canned generations so the loop can be tested without weights."""

    call_format = LFM2

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen: list[dict[str, str]] = []

    def generate(self, messages, **kwargs):
        self.seen = list(messages)
        return Generation(
            text=self.outputs.pop(0), prompt_tokens=10, generated_tokens=5, seconds=0.1
        )


def test_agent_feeds_tool_result_back_and_answers(tmp_path):
    spec = AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",))
    model = ScriptedModel(
        [
            "<|tool_call_start|>[calculator(expression='6 * 7')]<|tool_call_end|>",
            "The answer is 42.",
        ]
    )

    result = run_agent(model, spec, "What is 6 times 7?", build_tools(tmp_path))

    assert isinstance(result, AgentResult)
    assert [(c.name, c.result) for c in result.calls] == [("calculator", "42")]
    assert result.answer == "The answer is 42."
    assert not result.hit_step_limit
    assert [step.index for step in result.steps] == [0, 1]
    assert result.generated_tokens == 10


def test_agent_reports_tool_errors_to_the_model(tmp_path):
    spec = AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",))
    model = ScriptedModel(
        ["<|tool_call_start|>[calculator(expression='nope')]", "I could not compute that."]
    )

    result = run_agent(model, spec, "compute nope", build_tools(tmp_path))

    assert result.calls[0].result.startswith("error:")
    assert model.seen[-1]["role"] == "tool"
    assert model.seen[-1]["content"].startswith("error:")


def test_agent_without_tools_answers_in_one_step(tmp_path):
    spec = AgentSpec(name="chat", system_prompt="be brief")
    model = ScriptedModel(["calculator('1+1')"])

    result = run_agent(model, spec, "hi", build_tools(tmp_path))

    assert result.calls == []
    assert result.answer == "calculator('1+1')"


def test_step_limit_forces_a_final_answer(tmp_path):
    spec = AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",), max_steps=2)
    model = ScriptedModel(["calculator('1+1')", "calculator('2+2')", "I gave up: 4."])

    result = run_agent(model, spec, "loop forever", build_tools(tmp_path))

    assert result.hit_step_limit
    assert result.answer == "I gave up: 4."


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "agents.yaml"
    path.write_text("agents:\n  chat:\n    system_prompt: hi\n    temprature: 0.9\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_agents(path)


def test_shipped_config_loads():
    agents = load_agents(config_path("agents.yaml"))
    assert "chat" in agents
    assert agents["calc"].tools == ("calculator",)


def test_history_is_placed_between_system_and_prompt(tmp_path):
    spec = AgentSpec(name="chat", system_prompt="be brief")
    model = ScriptedModel(["fine"])
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    run_agent(model, spec, "and now?", build_tools(tmp_path), history=history)

    assert [m["role"] for m in model.seen] == ["system", "user", "assistant", "user"]
    assert model.seen[-1]["content"] == "and now?"


def test_step_callback_fires_as_each_step_finishes(tmp_path):
    spec = AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",))
    model = ScriptedModel(["calculator('6 * 7')", "42"])
    seen = []

    run_agent(model, spec, "q", build_tools(tmp_path), on_step=seen.append)

    assert [(s.call.name, s.call.result) for s in seen if s.call] == [("calculator", "42")]
    assert seen[-1].call is None and seen[-1].generation == "42"


def test_steps_record_the_raw_generation_and_cost(tmp_path):
    spec = AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",))
    model = ScriptedModel(["calculator('6 * 7')", "42"])

    result = run_agent(model, spec, "q", build_tools(tmp_path))

    assert result.steps[0].generation == "calculator('6 * 7')"
    assert result.steps[0].call.result == "42"
    assert result.steps[1].call is None
    assert result.seconds == pytest.approx(0.2, abs=0.05)


def test_forced_first_call_is_recorded_on_the_step(tmp_path):
    spec = AgentSpec(name="calc", system_prompt="s", tools=("calculator",), force_first_call=True)
    model = ScriptedModel(["calculator('1+1')", "2"])

    result = run_agent(model, spec, "q", build_tools(tmp_path))

    assert result.steps[0].forced
    assert not result.steps[1].forced


def test_tool_markup_left_in_an_answer_is_not_printed(tmp_path):
    """Qwen2.5-Coder ends turns by repeating the call instead of writing prose."""
    spec = AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",))
    model = ScriptedModel(
        [
            "calculator('6 * 7')",
            '{"name": "calculator", "arguments": {"expression": "42"}}',
        ]
    )

    result = run_agent(model, spec, "what is 6 times 7", build_tools(tmp_path))

    assert result.answer == "42"
    assert "arguments" not in result.answer


def test_an_answer_that_strips_to_nothing_falls_back_to_the_tool(tmp_path):
    spec = AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",))
    model = ScriptedModel(["calculator('6 * 7')", "<|tool_call_end|>"])

    assert run_agent(model, spec, "q", build_tools(tmp_path)).answer == "42"


def test_prose_is_left_alone(tmp_path):
    spec = AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",))
    model = ScriptedModel(["calculator('6 * 7')", "The answer is 42."])

    assert run_agent(model, spec, "q", build_tools(tmp_path)).answer == "The answer is 42."
