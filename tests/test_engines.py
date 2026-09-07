import pytest

from root.engines import (
    AUTO,
    DEFAULT_URLS,
    ENGINES,
    PREFERENCE,
    TRANSFORMERS,
    choose_engine,
    default_url,
    load_engine,
)


def test_every_served_engine_has_a_default_url():
    served = set(ENGINES) - {TRANSFORMERS, AUTO}
    assert served == set(DEFAULT_URLS)


def test_preference_covers_every_real_engine_fastest_first():
    assert set(PREFERENCE) == set(ENGINES) - {AUTO}
    assert PREFERENCE[0] == "ollama"
    assert PREFERENCE[-1] == TRANSFORMERS


def test_auto_falls_back_to_the_local_engine_when_nothing_is_serving(monkeypatch):
    monkeypatch.setattr("root.engines.reachable", lambda *a, **k: False)
    assert choose_engine("some/model") == TRANSFORMERS


def test_auto_takes_the_fastest_engine_that_is_serving(monkeypatch):
    monkeypatch.setattr("root.engines.reachable", lambda engine, *a, **k: engine == "vllm")
    assert choose_engine("some/model") == "vllm"


def test_a_url_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("ROOT_OLLAMA_URL", "http://box:9999")
    assert default_url("ollama") == "http://box:9999"
    assert default_url("vllm") == DEFAULT_URLS["vllm"]


def test_an_unknown_engine_is_rejected_before_anything_loads():
    with pytest.raises(ValueError, match="unknown engine"):
        load_engine("some/model", "tensorrt")


def test_a_freeform_parameter_disables_the_grammar():
    """The grammar covers the whole generation, so one unconstrainable branch
    means the turn cannot be constrained at all."""
    from pathlib import Path

    from root.grammar import signatures_from
    from root.tools import build_tools

    tools = build_tools(Path("."))
    assert signatures_from([tools["calculator"].schema()]) == [("calculator", ("expression",))]
    assert signatures_from([tools["run_python"].schema()]) is None
    assert signatures_from([tools["calculator"].schema(), tools["run_python"].schema()]) is None


def test_no_tools_means_no_grammar():
    from root.grammar import signatures_from

    assert signatures_from([]) == []
    assert signatures_from(None) == []


def test_tokenspeed_is_a_served_openai_engine():
    """It exposes /v1/completions, so it needs no code of its own."""
    assert "tokenspeed" in ENGINES
    assert DEFAULT_URLS["tokenspeed"].endswith(":8000")


def test_tokenspeed_is_preferred_over_the_local_engine():
    assert PREFERENCE.index("tokenspeed") < PREFERENCE.index(TRANSFORMERS)


def test_both_engines_take_the_same_generate_arguments():
    """run_agent calls generate the same way whichever engine is underneath.

    A parameter added to one and not the other is invisible to every test that
    uses a scripted model, and breaks all four served engines at runtime.
    """
    from inspect import signature

    from root.backend import LocalModel
    from root.engines import ServedModel

    local = set(signature(LocalModel.generate).parameters)
    served = set(signature(ServedModel.generate).parameters)

    assert local == served, f"only in one: {local ^ served}"


def test_an_ollama_tag_is_guessed_from_a_hugging_face_id():
    from root.engines import ollama_tag

    assert ollama_tag("Qwen/Qwen2.5-Coder-0.5B-Instruct") == "qwen2.5-coder:0.5b"
    assert ollama_tag("Qwen/Qwen3-0.6B") == "qwen3:0.6b"
    assert ollama_tag("LiquidAI/LFM2.5-350M") == "lfm2.5:350m"


def test_names_match_regardless_of_punctuation():
    """`qwen2.5-coder` failed to match `Qwen2.5-Coder-0.5B-Instruct` while it
    was installed, because the dot was stripped from one side only."""
    from root.engines import _flatten

    assert _flatten("qwen2.5-coder") in _flatten("Qwen2.5-Coder-0.5B-Instruct")
    assert _flatten("qwen3") in _flatten("Qwen3-0.6B")


def test_probing_never_pulls():
    """`auto` asks every engine whether it is serving the model. A question
    must not download two gigabytes or print a failed pull at every startup."""
    import inspect

    from root import engines

    source = inspect.getsource(engines.reachable)
    assert "allow_pull=False" in source
