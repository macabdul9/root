from pathlib import Path

import pytest

from root.agent import AgentResult, Step, ToolCall
from root.config import config_path, load_agents
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

    assert {path.name for path in written} == {
        "agents.yaml",
        "evals.yaml",
        "mcp.yaml",
        "models.yaml",
        "workspace",
    }
    assert load_agents(tmp_path / "configs" / "agents.yaml")["chat"].name == "chat"


def test_init_brings_the_eval_fixture_with_it(tmp_path):
    """Copying only the YAML leaves evals.yaml pointing at a workspace that is
    not there, and every file case then fails as though the model got it wrong."""
    from root.config import copy_defaults

    copy_defaults(tmp_path / "configs")

    assert (tmp_path / "configs" / "workspace" / "pipeline.py").is_file()


def test_a_case_workspace_falls_back_to_the_packaged_fixture(tmp_path):
    from root.config import PACKAGED_CONFIGS
    from root.evaluate import case_workspace

    evals = tmp_path / "evals.yaml"
    evals.write_text("cases: []\n")

    assert case_workspace("workspace", evals, tmp_path) == PACKAGED_CONFIGS / "workspace"

    beside = tmp_path / "workspace"
    beside.mkdir()
    assert case_workspace("workspace", evals, tmp_path) == beside


def test_init_leaves_existing_files_alone(tmp_path):
    from root.config import copy_defaults

    target = tmp_path / "configs"
    target.mkdir()
    (target / "agents.yaml").write_text("mine\n")

    written = copy_defaults(target)

    assert "agents.yaml" not in {path.name for path in written}
    assert (target / "agents.yaml").read_text() == "mine\n"


def test_scores_group_by_agent_in_first_seen_order():
    from root.evals import by_agent

    scores = [
        score_case(EvalCase(agent="calc", prompt="q"), result()),
        score_case(EvalCase(agent="chat", prompt="q"), result()),
        score_case(EvalCase(agent="calc", prompt="q"), result()),
    ]
    grouped = by_agent(scores)

    assert list(grouped) == ["calc", "chat"]
    assert len(grouped["calc"]) == 2


def test_a_negative_expectation_fails_when_the_phrase_appears():
    case = EvalCase(agent="chat", prompt="q", expect_answer_excludes=("has been created",))

    assert score_case(case, result("I cannot create files.")).answer_ok
    assert not score_case(case, result("The file has been created.")).answer_ok


def test_greedy_decoding_omits_the_sampling_knobs():
    from root.config import Decoding

    assert Decoding(temperature=0.0).as_generate_kwargs() == {
        "max_new_tokens": 256,
        "do_sample": False,
    }


def test_sampling_passes_only_the_knobs_that_are_on():
    from root.config import Decoding

    kwargs = Decoding(temperature=0.8, top_k=50).as_generate_kwargs()

    assert kwargs["do_sample"] and kwargs["top_k"] == 50
    assert "min_p" not in kwargs
    assert "repetition_penalty" not in kwargs


def test_an_agent_exposes_its_decoding_settings():
    from root.config import AgentSpec

    spec = AgentSpec(name="x", system_prompt="y", temperature=0.0, top_k=20, seed=3)

    assert spec.decoding.samples is False
    assert spec.decoding.top_k == 20
    assert spec.decoding.seed == 3


def test_an_invalid_knob_names_the_agent():
    from root.config import AgentSpec

    with pytest.raises(ValueError, match="agent x: repetition_penalty"):
        AgentSpec(name="x", system_prompt="y", repetition_penalty=0)


def test_registering_writes_a_local_config_from_the_packaged_one(tmp_path):
    from root.config import load_models, register_model

    target = tmp_path / "configs" / "models.yaml"
    choice = register_model("vendor/Tiny-1B", note="1B params", path=target)

    assert choice.alias == "tiny-1b"
    registered = load_models(target)
    assert registered["tiny-1b"].id == "vendor/Tiny-1B"
    assert registered["tiny-1b"].note == "1B params"
    # The packaged defaults come along, so the list does not shrink.
    assert "lfm2-350m" in registered


def test_registering_keeps_the_comments_in_the_file(tmp_path):
    from root.config import register_model

    target = tmp_path / "configs" / "models.yaml"
    register_model("vendor/Tiny-1B", path=target)

    assert "# Models the /model command offers" in target.read_text()


def test_an_explicit_alias_is_used(tmp_path):
    from root.config import register_model

    choice = register_model("vendor/Tiny-1B", alias="tiny", path=tmp_path / "models.yaml")
    assert choice.alias == "tiny"


def test_registering_the_same_alias_twice_is_refused(tmp_path):
    from root.config import register_model

    target = tmp_path / "models.yaml"
    register_model("vendor/Tiny-1B", path=target)
    with pytest.raises(ValueError, match="already registered"):
        register_model("vendor/Tiny-1B", path=target)


def test_trust_remote_code_survives_a_round_trip(tmp_path):
    from root.config import load_models, register_model

    target = tmp_path / "models.yaml"
    register_model("vendor/Custom", trust_remote_code=True, path=target)
    assert load_models(target)["custom"].trust_remote_code


def test_unregistering_removes_the_entry(tmp_path):
    from root.config import load_models, register_model, unregister_model

    target = tmp_path / "models.yaml"
    register_model("vendor/Tiny-1B", path=target)

    assert unregister_model("tiny-1b", path=target) == "vendor/Tiny-1B"
    assert "tiny-1b" not in load_models(target)


def test_unregistering_something_absent_is_reported(tmp_path):
    from root.config import register_model, unregister_model

    target = tmp_path / "models.yaml"
    register_model("vendor/Tiny-1B", path=target)
    with pytest.raises(ValueError, match="not registered"):
        unregister_model("nothing", path=target)


def test_removing_the_default_moves_it_rather_than_leaving_it_dangling(tmp_path):
    from root.config import load_default_model, register_model, unregister_model

    target = tmp_path / "models.yaml"
    register_model("vendor/Tiny-1B", path=target)
    unregister_model("lfm2-350m", path=target)

    # The old default is gone, so load_default_model must not point at it.
    assert load_default_model(target) != "LiquidAI/LFM2.5-350M"


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("LiquidAI/LFM2.5-350M", "lfm2.5-350m"),
        ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
        # The tuning suffix is dropped: it does not distinguish one entry from
        # another, and qwen2.5-coder-0.5b-instruct is a lot to type.
        ("meta-llama/Llama-3.2-1B-Instruct", "llama-3.2-1b"),
        ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2.5-coder-0.5b"),
        ("someone/weird__name!!", "weird-name"),
    ],
)
def test_aliases_are_derived_from_the_repository_name(model_id, expected):
    from root.config import alias_for

    assert alias_for(model_id) == expected


def test_a_model_over_the_limit_is_refused():
    """features.txt: never run a model bigger than 1B in process."""
    from root.config import check_size

    with pytest.raises(ValueError, match="over the 1B limit"):
        check_size("meta-llama/Llama-3.1-8B", 8_030_000_000)


def test_a_model_under_the_limit_is_allowed():
    from root.config import check_size

    check_size("vendor/small", 500_000_000)


def test_an_unknown_size_is_allowed_rather_than_guessed():
    """Refusing every model with incomplete metadata is worse than allowing it."""
    from root.config import check_size

    check_size("vendor/unpublished", None)


def test_registering_an_oversized_model_is_refused(tmp_path, monkeypatch):
    from root import config as config_module

    monkeypatch.setattr(config_module, "hub_parameters", lambda _id: 8_030_000_000)
    with pytest.raises(ValueError, match="over the 1B limit"):
        config_module.check_size("vendor/huge")
