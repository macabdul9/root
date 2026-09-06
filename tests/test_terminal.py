from pathlib import Path

import pytest

from root.agent import AgentResult, Step
from root.config import AgentSpec
from root.terminal import HISTORY_TURNS, Session, run_command
from root.tools import build_tools


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
    assert listing.splitlines()[0].startswith("* chat")


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
    assert run_command(session, "/trace") == "trace: on"
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
