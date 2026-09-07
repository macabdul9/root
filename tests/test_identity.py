from pathlib import Path

from root.agent import compose_system, run_agent
from root.config import AgentSpec, config_path, load_agents
from root.identity import AUTHOR, NAME, describe
from root.tools import build_tools

AGENTS = load_agents(config_path("agents.yaml"))


def tools_for(*names):
    everything = build_tools(Path("."))
    return {name: everything[name] for name in names}


def test_identity_names_root_its_author_and_the_loaded_model():
    said = describe("LiquidAI/LFM2.5-350M", tools_for("calculator"), AGENTS)

    assert NAME in said
    assert AUTHOR in said
    assert "LiquidAI/LFM2.5-350M" in said
    assert "not scary" in said


def test_identity_follows_the_model_that_is_loaded():
    assert "Qwen/Qwen3-0.6B" in describe("Qwen/Qwen3-0.6B", tools_for("calculator"), AGENTS)


def test_identity_lists_only_the_tools_that_exist():
    """Built from the registry, so it cannot claim an ability that was removed."""
    said = describe("m", tools_for("calculator"), AGENTS)

    assert "arithmetic" in said
    assert "PDF" not in said


def test_a_question_about_root_never_reaches_the_model():
    """A model this size invents its own provenance, so it is not asked."""

    class Exploding:
        model_id = "LiquidAI/LFM2.5-350M"
        call_format = None

        def generate(self, *args, **kwargs):
            raise AssertionError("the model was asked who it is")

    spec = AgentSpec(name="chat", system_prompt="be brief")
    result = run_agent(Exploding(), spec, "who are you?", tools_for("calculator"), agents=AGENTS)

    assert NAME in result.answer
    assert result.calls == []


def test_the_shipped_preamble_is_empty():
    """It measurably hurts a 350M model: python 13/20 to 9/20, search 9/10 to
    5/10. The text is kept in the config, commented out, for larger models."""
    assert all(spec.preamble == "" for spec in AGENTS.values())


def test_a_preamble_is_prepended_to_the_agent_prompt():
    from dataclasses import replace

    spec = replace(AGENTS["calc"], preamble="You are root on {{model}}.")
    composed = compose_system(spec, tools_for("calculator"), "vendor/tiny")

    assert composed.startswith("You are root on vendor/tiny.")
    assert AGENTS["calc"].system_prompt.strip() in composed


def test_a_preamble_substitutes_what_is_loaded():
    from dataclasses import replace

    template = "{{agent}} on {{model}} with {{tools}} on {{date}}"
    spec = replace(AGENTS["calc"], preamble=template)
    composed = compose_system(spec, tools_for("calculator"), "vendor/tiny")

    assert composed.startswith("calc on vendor/tiny with calculator on 20")
    assert "{{" not in composed


def test_substitution_leaves_braces_in_a_prompt_alone():
    """Plain replacement, not str.format: an agent that shows JSON must not
    have its braces treated as fields."""
    from dataclasses import replace

    spec = replace(AGENTS["extract"], preamble="on {{model}}", system_prompt='reply {"a": 1}')
    assert '{"a": 1}' in compose_system(spec, {}, "m")


def test_an_agent_without_a_preamble_is_unchanged():
    spec = AgentSpec(name="bare", system_prompt="just this")
    assert compose_system(spec, {}, "m") == "just this"
