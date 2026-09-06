import pytest

from root.agent import AgentResult, Step, ToolCall
from root.evals import EvalCase, accuracy, load_cases, score_case


def result(answer="", calls=()):
    steps = [
        Step(
            index=i,
            generation="",
            seconds=0.0,
            prompt_tokens=0,
            generated_tokens=0,
            call=call,
        )
        for i, call in enumerate(calls)
    ]
    return AgentResult(agent="a", prompt="p", answer=answer, steps=steps)


def test_expected_tool_must_be_called():
    case = EvalCase(agent="calc", prompt="2+2", expect_tool="calculator")
    assert not score_case(case, result("4")).tool_ok
    assert score_case(case, result("4", [ToolCall("calculator", "2+2", "4")])).tool_ok


def test_expect_tool_none_requires_a_toolless_answer():
    case = EvalCase(agent="chat", prompt="hi", expect_tool="none")
    assert score_case(case, result("hello")).tool_ok
    assert not score_case(case, result("hello", [ToolCall("grep", "x", "y")])).tool_ok


def test_argument_check_applies_to_the_expected_tool():
    case = EvalCase(
        agent="files",
        prompt="q",
        expect_tool="read_file",
        expect_argument_contains="pyproject.toml",
    )
    calls = [ToolCall("read_file", "PyProject.toml", "...")]
    assert score_case(case, result("3.10", calls)).argument_ok
    assert not score_case(
        case, result("3.10", [ToolCall("read_file", "README.md", "")])
    ).argument_ok


def test_answer_check_needs_every_substring():
    case = EvalCase(agent="extract", prompt="q", expect_answer_contains=("Dana", "412.50"))
    assert score_case(case, result('{"name": "Dana", "amount": "412.50"}')).answer_ok
    assert not score_case(case, result('{"name": "Dana"}')).answer_ok


def test_argument_expectation_without_a_tool_is_rejected():
    with pytest.raises(ValueError, match="expect_argument_contains"):
        EvalCase(agent="files", prompt="q", expect_argument_contains="x")


def test_accuracy_separates_tool_choice_from_answer():
    cases = [EvalCase(agent="calc", prompt="q", expect_tool="calculator")]
    scores = [score_case(cases[0], result("wrong", [ToolCall("calculator", "1", "1")]))]
    assert accuracy(scores) == {"tool": 1.0, "answer": 1.0, "overall": 1.0}


def test_shipped_eval_set_loads(tmp_path):
    from pathlib import Path

    cases = load_cases(Path("configs/evals.yaml"))
    assert {case.agent for case in cases} >= {"calc", "files", "python", "search"}
