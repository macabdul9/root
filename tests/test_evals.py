from pathlib import Path

import pytest

from root.agent import AgentResult, Step, ToolCall
from root.config import config_path
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

    cases = load_cases(config_path("evals.yaml"))
    assert {case.agent for case in cases} >= {"calc", "files", "python", "search"}


def test_default_model_comes_from_the_config(tmp_path):
    from root.config import load_default_model

    path = tmp_path / "models.yaml"
    path.write_text("default: b\nmodels:\n  a:\n    id: vendor/a\n  b:\n    id: vendor/b\n")
    assert load_default_model(path) == "vendor/b"


def test_default_model_falls_back_when_the_alias_is_unknown(tmp_path):
    from root.config import FALLBACK_MODEL, load_default_model

    path = tmp_path / "models.yaml"
    path.write_text("default: missing\nmodels:\n  a:\n    id: vendor/a\n")
    assert load_default_model(path) == FALLBACK_MODEL


def test_default_model_falls_back_without_a_config(tmp_path):
    from root.config import FALLBACK_MODEL, load_default_model

    assert load_default_model(tmp_path / "absent.yaml") == FALLBACK_MODEL


def test_the_shipped_default_is_lfm2_350m():

    from root.config import load_default_model

    assert load_default_model(config_path("models.yaml")) == "LiquidAI/LFM2.5-350M"


def test_a_local_config_directory_overrides_the_packaged_one(tmp_path, monkeypatch):
    from root import config as config_module

    monkeypatch.chdir(tmp_path)
    assert (
        config_module.config_path("agents.yaml") == config_module.PACKAGED_CONFIGS / "agents.yaml"
    )

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "agents.yaml").write_text("agents: {}\n")
    assert config_module.config_path("agents.yaml") == Path("configs/agents.yaml")


def test_an_explicit_path_wins_over_both(tmp_path):
    from root.config import config_path

    assert config_path("agents.yaml", tmp_path / "mine.yaml") == tmp_path / "mine.yaml"


def test_init_copies_the_packaged_configs(tmp_path):
    from root.config import copy_defaults

    written = copy_defaults(tmp_path / "configs")

    assert {path.name for path in written} == {"agents.yaml", "evals.yaml", "models.yaml"}
    assert (tmp_path / "configs" / "agents.yaml").read_text().startswith("defaults:")


def test_init_leaves_existing_files_alone(tmp_path):
    from root.config import copy_defaults

    target = tmp_path / "configs"
    target.mkdir()
    (target / "agents.yaml").write_text("mine\n")

    written = copy_defaults(target)

    assert "agents.yaml" not in {path.name for path in written}
    assert (target / "agents.yaml").read_text() == "mine\n"
