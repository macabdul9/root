from pathlib import Path
from types import SimpleNamespace

import pytest

from root import terminal
from root.agent import AgentResult, Step
from root.config import AgentSpec, ModelChoice
from root.formats import GENERIC
from root.terminal import HISTORY_TURNS, Session, run_command
from root.tools import build_tools


def _record(seen, choice):
    seen.append(choice)
    return SimpleNamespace(model_id=choice.id, call_format=GENERIC, unload=lambda: None)


@pytest.fixture
def session(tmp_path):
    agents = {
        "chat": AgentSpec(name="chat", system_prompt="be brief"),
        "calc": AgentSpec(name="calc", system_prompt="solve it", tools=("calculator",)),
    }
    return Session(agents=agents, agent="chat", tools=build_tools(tmp_path), workspace=tmp_path)


def test_switching_agent_clears_the_conversation(session):
    session.remember("hi", "hello")
    assert run_command(session, "/agent calc") == "agent: calc"
    assert session.agent == "calc"
    assert session.history == []


def test_switching_to_an_unknown_agent_is_reported(session):
    assert "unknown agent" in run_command(session, "/agent nope")
    assert session.agent == "chat"


def test_quit_stops_the_loop(session):
    run_command(session, "/quit")
    assert not session.running


def test_agents_marks_the_current_one(session):
    listing = run_command(session, "/agents")
    assert "* chat" in listing
    assert listing.splitlines()[0].startswith("  auto")


def test_auto_is_an_accepted_agent(session):
    assert run_command(session, "/agent auto") == "agent: auto"
    assert session.agent == "auto"


def test_auto_resolves_the_agent_from_the_prompt(session):
    run_command(session, "/agent auto")
    spec, rule = session.resolve("What is 17 times 23?")
    assert spec.name == "calc"
    assert rule == "arithmetic"


def test_a_named_agent_is_used_verbatim(session):
    spec, rule = session.resolve("What is 17 times 23?")
    assert spec.name == "chat"
    assert rule is None


def test_tools_explains_auto(session):
    run_command(session, "/agent auto")
    assert "picks an agent per prompt" in run_command(session, "/tools")


def test_routes_lists_the_rules(session):
    listing = run_command(session, "/routes")
    assert "arithmetic" in listing
    assert "calc" in listing


def test_tools_reports_a_toolless_agent(session):
    assert "without tools" in run_command(session, "/tools")


def test_tools_lists_what_the_agent_may_call(session):
    run_command(session, "/agent calc")
    assert run_command(session, "/tools").startswith("calculator(expression)")


def test_workspace_change_rebuilds_the_tools(session, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "note.txt").write_text("visible\n")

    assert str(other) in run_command(session, f"/workspace {other}")
    assert session.tools["read_file"].run("note.txt") == "visible\n"


def test_workspace_rejects_a_missing_directory(session):
    assert run_command(session, "/workspace /no/such/place").startswith("not a directory")


def test_unknown_command_suggests_help(session):
    assert "/help" in run_command(session, "/nonsense")


def test_history_keeps_only_the_recent_turns(session):
    for i in range(HISTORY_TURNS + 3):
        session.remember(f"q{i}", f"a{i}")

    assert len(session.history) == HISTORY_TURNS * 2
    assert session.history[0]["content"] == f"q{HISTORY_TURNS + 3 - HISTORY_TURNS}"
    assert session.history[-1]["content"] == f"a{HISTORY_TURNS + 2}"


def test_history_records_the_answer_not_the_tool_calls(session):
    session.remember("what is 6*7", "42")
    assert session.history == [
        {"role": "user", "content": "what is 6*7"},
        {"role": "assistant", "content": "42"},
    ]


def test_workspace_without_an_argument_just_reports(session, tmp_path):
    assert run_command(session, "/workspace") == f"workspace: {Path(tmp_path)}"


def test_trace_level_is_shown_and_set(session):
    assert run_command(session, "/trace").startswith("trace: on")
    assert run_command(session, "/trace full") == "trace: full"
    assert session.trace == "full"


def test_invalid_trace_level_is_rejected(session):
    assert "must be one of" in run_command(session, "/trace sideways")
    assert session.trace == "on"


def test_last_needs_a_previous_turn(session):
    assert run_command(session, "/last") == "nothing to replay yet"


def test_last_replays_the_previous_turn(session):
    step = Step(index=0, generation="hello", seconds=0.2, prompt_tokens=5, generated_tokens=3)
    session.last = AgentResult(agent="chat", prompt="hi", answer="hello", steps=[step])
    replay = run_command(session, "/last")
    assert "raw: hello" in replay
    assert replay.endswith("hello")


def test_model_listing_marks_the_loaded_one(session):
    session.models = {
        "lfm2-350m": ModelChoice("lfm2-350m", "LiquidAI/LFM2.5-350M", "the default"),
        "qwen3-06b": ModelChoice("qwen3-06b", "Qwen/Qwen3-0.6B", "JSON tool calls"),
    }
    session.model = SimpleNamespace(model_id="Qwen/Qwen3-0.6B")

    listing = run_command(session, "/model")

    assert "  lfm2-350m" in listing
    assert "* qwen3-06b" in listing


def test_an_unlisted_model_still_shows_as_current(session):
    session.model = SimpleNamespace(model_id="someone/custom-model")
    assert "someone/custom-model" in run_command(session, "/model")


def test_switching_model_resolves_an_alias_and_clears_history(session, monkeypatch):
    session.models = {"qwen3-06b": ModelChoice("qwen3-06b", "Qwen/Qwen3-0.6B")}
    session.model = SimpleNamespace(model_id="LiquidAI/LFM2.5-350M", unload=lambda: None)
    session.remember("hi", "hello")
    loaded = SimpleNamespace(model_id="Qwen/Qwen3-0.6B", call_format=GENERIC)
    monkeypatch.setattr(terminal, "load_model", lambda choice, device, engine=None: loaded)

    message = run_command(session, "/model qwen3-06b")

    assert session.model is loaded
    assert "Qwen/Qwen3-0.6B" in message
    assert session.history == []


def test_switching_to_the_loaded_model_does_nothing(session):
    session.model = SimpleNamespace(model_id="LiquidAI/LFM2.5-350M")
    assert run_command(session, "/model LiquidAI/LFM2.5-350M").startswith("already on")


def test_an_unknown_model_name_is_passed_through_as_an_id(session, monkeypatch):
    session.model = SimpleNamespace(model_id="a/b", unload=lambda: None)
    seen = []
    monkeypatch.setattr(
        terminal, "load_model", lambda choice, device, engine=None: _record(seen, choice)
    )

    run_command(session, "/model some/other-model")

    assert seen[0].id == "some/other-model"
    assert not seen[0].trust_remote_code


def test_remote_code_is_only_used_when_the_entry_asks_for_it(session, monkeypatch):
    session.models = {
        "safe": ModelChoice("safe", "vendor/safe"),
        "custom": ModelChoice("custom", "vendor/custom", trust_remote_code=True),
    }
    session.model = SimpleNamespace(model_id="a/b", unload=lambda: None)
    seen = []
    monkeypatch.setattr(
        terminal, "load_model", lambda choice, device, engine=None: _record(seen, choice)
    )

    run_command(session, "/model safe")
    run_command(session, "/model custom")

    assert [choice.trust_remote_code for choice in seen] == [False, True]


def test_bare_agent_lists_instead_of_erroring(session):
    listing = run_command(session, "/agent")
    assert "unknown agent" not in listing
    assert "/agent NAME" in listing
    assert session.agent == "chat"


def test_bare_trace_says_how_to_change_it(session):
    assert run_command(session, "/trace") == "trace: on · change with /trace off | full"


def test_model_listing_says_how_to_load_one(session):
    session.model = SimpleNamespace(model_id="a/b")
    assert "/model ALIAS" in run_command(session, "/model")


def test_decoding_shows_the_current_agent_settings(session):
    shown = run_command(session, "/decoding")
    assert shown.startswith("chat: temperature=")
    assert "max_new_tokens=" in shown


def test_decoding_sets_several_values_at_once(session):
    run_command(session, "/decoding temperature=0.9 top_k=40")
    assert session.spec.decoding.temperature == 0.9
    assert session.spec.decoding.top_k == 40


def test_decoding_rejects_an_out_of_range_value(session):
    assert "top_p must be in" in run_command(session, "/decoding top_p=3")
    assert session.spec.decoding.top_p == 0.9


def test_decoding_rejects_an_unknown_setting(session):
    assert "unknown setting" in run_command(session, "/decoding warmth=2")


def test_decoding_rejects_a_non_number(session):
    assert "needs a number" in run_command(session, "/decoding temperature=warm")


def test_decoding_rejects_a_bare_word(session):
    assert "expected name=value" in run_command(session, "/decoding hot")


def test_stream_toggles(session):
    assert run_command(session, "/stream") == "stream: on"
    assert run_command(session, "/stream off").startswith("stream: off")
    assert not session.stream
    run_command(session, "/stream on")
    assert session.stream


def test_stream_rejects_anything_else(session):
    assert run_command(session, "/stream fast") == "usage: /stream on|off"


def test_inference_shows_the_current_engine(session):
    assert run_command(session, "/inference").startswith("inference: transformers")


def test_inference_rejects_an_unknown_engine(session):
    assert "unknown engine" in run_command(session, "/inference tensorrt")
    assert session.engine == "transformers"


def test_switching_to_the_current_engine_does_nothing(session):
    assert run_command(session, "/inference transformers") == "already on transformers"


def test_a_failed_engine_switch_keeps_the_old_model(session, monkeypatch):
    from root.engines import EngineUnavailable

    original = SimpleNamespace(model_id="a/b", unload=lambda: None)
    session.model = original

    def refuse(choice, device, engine):
        raise EngineUnavailable("ollama at http://127.0.0.1:11434 is not reachable")

    monkeypatch.setattr(terminal, "load_model", refuse)

    message = run_command(session, "/inference ollama")

    assert "not reachable" in message
    assert session.engine == "transformers"
    assert session.model is original


def test_load_model_accepts_the_arguments_its_callers_pass():
    """The commands monkeypatch load_model, so only this checks the real one.

    A signature that drifted from its callers shipped once already: /model and
    /inference both raised TypeError at runtime while every test passed.
    """
    from inspect import signature

    from root.config import ModelChoice

    signature(terminal.load_model).bind(ModelChoice("a", "a/b"), None, "transformers")


def test_engine_is_an_alias_for_inference(session):
    assert run_command(session, "/engine") == run_command(session, "/inference")


def test_a_pinned_agent_is_flagged_when_auto_would_differ(session):
    """`/agent python` then `save it to notes.txt` runs code instead of saving,
    and nothing said why."""
    from root.route import choose_agent

    session.agent = "python"
    elsewhere = choose_agent("save it to a file called notes.txt", set(session.agents) | {"write"})

    assert elsewhere.matched
    assert elsewhere.agent != session.agent
