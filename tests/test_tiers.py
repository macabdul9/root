from __future__ import annotations

import pytest

from root.config import ModelChoice
from root.tiers import Ladder, Router, Rung, Rungs, load_ladder, score_to_tier

MODELS = {
    "small-one": ModelChoice(alias="small-one", id="org/Small-1"),
    "mid-one": ModelChoice(alias="mid-one", id="org/Mid-1"),
    "big-one": ModelChoice(alias="big-one", id="org/Big-1"),
}


class FakeModel:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True


def full_ladder() -> Ladder:
    return load_ladder(
        {
            "router": "mid-one",
            "tiers": {
                "small": "small-one",
                "medium": "mid-one",
                "large": {"model": "big-one", "engine": "vllm", "url": "http://host:8000"},
            },
        }
    )


def test_a_tier_can_be_an_alias_or_a_mapping():
    ladder = full_ladder()
    assert ladder.rungs["small"] == Rung(tier="small", model="small-one")
    assert ladder.rungs["large"].engine == "vllm"
    assert ladder.rungs["large"].url == "http://host:8000"


def test_an_unknown_tier_name_is_rejected():
    with pytest.raises(ValueError, match="unknown tiers"):
        load_ladder({"tiers": {"enormous": "big-one"}})


def test_a_tier_with_no_model_is_rejected():
    with pytest.raises(ValueError, match="no model named"):
        load_ladder({"tiers": {"small": {"engine": "vllm"}}})


def test_an_empty_tier_is_left_out_rather_than_being_an_error():
    ladder = load_ladder({"tiers": {"small": "small-one", "medium": None}})
    assert ladder.routed == ["small"]


def test_the_top_tier_is_configurable_but_never_routed_to():
    ladder = load_ladder(
        {"router": "mid-one", "tiers": {"small": "small-one", "xlarge": "big-one"}}
    )
    assert "xlarge" in ladder.rungs
    assert ladder.routed == ["small"]
    # One routable rung is nothing to choose between.
    assert not ladder.usable


def test_routing_needs_a_router_and_two_rungs():
    assert full_ladder().usable
    assert not load_ladder({"tiers": {"small": "small-one", "medium": "mid-one"}}).usable


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "small"),
        (1.00, "small"),
        (1.01, "medium"),
        (1.39, "medium"),
        (1.40, "large"),
        (3.0, "large"),
    ],
)
def test_a_score_lands_on_the_tier_its_band_covers(score, expected):
    assert score_to_tier(score, ["small", "medium", "large"]) == expected


def test_a_two_rung_ladder_splits_at_the_first_boundary():
    assert score_to_tier(0.9, ["small", "large"]) == "small"
    assert score_to_tier(2.5, ["small", "large"]) == "large"


def test_scoring_an_empty_ladder_is_an_error():
    with pytest.raises(ValueError, match="no tiers configured"):
        score_to_tier(1.0, [])


def test_a_rung_is_loaded_once_and_then_reused(monkeypatch):
    calls = []

    def fake_load(model_id, engine, device=None, **kwargs):
        calls.append((model_id, engine, kwargs.get("base_url")))
        return FakeModel(model_id)

    monkeypatch.setattr("root.tiers.load_engine", fake_load)
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    first = rungs.open("large")
    assert rungs.open("large") is first
    assert calls == [("org/Big-1", "vllm", "http://host:8000")]


def test_an_unconfigured_tier_says_what_is_configured():
    rungs = Rungs(full_ladder(), MODELS)
    with pytest.raises(KeyError, match="no xlarge tier configured"):
        rungs.open("xlarge")


def test_the_session_model_is_adopted_rather_than_loaded_again(monkeypatch):
    monkeypatch.setattr(
        "root.tiers.load_engine",
        lambda *a, **k: pytest.fail("adopted rungs must not be loaded again"),
    )
    loaded = FakeModel("org/Small-1")
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    rungs.adopt(loaded)
    assert rungs.open("small") is loaded


def test_a_rung_served_elsewhere_is_not_adopted():
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    rungs.adopt(FakeModel("org/Big-1"))
    assert "large" not in rungs.loaded


def test_unloading_frees_opened_rungs_but_not_the_sessions_own_model(monkeypatch):
    monkeypatch.setattr("root.tiers.load_engine", lambda model_id, *a, **k: FakeModel(model_id))
    session_model = FakeModel("org/Small-1")
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    rungs.adopt(session_model)
    opened = rungs.open("medium")
    rungs.unload()
    assert opened.unloaded
    assert not session_model.unloaded


def test_picking_routes_the_prompt_to_the_tier_the_router_scored(monkeypatch):
    monkeypatch.setattr("root.tiers.load_engine", lambda model_id, *a, **k: FakeModel(model_id))
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    rungs.router = Router(
        model_id="org/Mid-1", model=None, tokenizer=None, device="cpu", letters={}, owned=False
    )
    monkeypatch.setattr(Router, "score", lambda self, prompt: 1.6)
    model, tier, score = rungs.pick("design a scheduler")
    assert (tier, score) == ("large", 1.6)
    assert model.model_id == "org/Big-1"


def test_a_borrowed_router_does_not_free_the_weights_it_borrowed():
    borrowed = Router(
        model_id="org/Mid-1", model=object(), tokenizer=None, device="cpu", letters={}, owned=False
    )
    borrowed.unload()
    assert borrowed.model is not None


def test_the_router_shares_the_weights_of_the_rung_that_is_the_same_model(monkeypatch):
    loads = []

    class LocalLike:
        def __init__(self, model_id):
            self.model_id = model_id
            self.model = object()
            self.tokenizer = object()
            self.device = "cpu"

    monkeypatch.setattr("root.tiers.LocalModel", LocalLike)
    monkeypatch.setattr("root.tiers._letter_ids", lambda tokenizer: {})
    monkeypatch.setattr(
        "root.tiers.Router.load", classmethod(lambda cls, *a, **k: pytest.fail("loaded twice"))
    )

    def fake_load(model_id, engine, device=None, **kwargs):
        loads.append(model_id)
        return LocalLike(model_id)

    monkeypatch.setattr("root.tiers.load_engine", fake_load)
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    router = rungs.ready()
    assert loads == ["org/Mid-1"]
    assert not router.owned
    assert router.model is rungs.loaded["medium"].model


def test_releasing_the_sessions_model_drops_the_rung_that_borrowed_it():
    loaded = FakeModel("org/Small-1")
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    rungs.adopt(loaded)
    rungs.release(loaded)
    assert not rungs.loaded
    assert not rungs.adopted


def test_releasing_the_sessions_model_drops_a_router_wrapped_around_it():
    class LocalLike:
        model_id = "org/Mid-1"
        model = object()

    loaded = LocalLike()
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    rungs.router = Router(
        model_id="org/Mid-1",
        model=loaded.model,
        tokenizer=None,
        device="cpu",
        letters={},
        owned=False,
    )
    rungs.release(loaded)
    assert rungs.router is None


def test_releasing_leaves_a_router_that_owns_its_own_weights_alone():
    rungs = Rungs(full_ladder(), MODELS, engine="transformers")
    rungs.router = Router(
        model_id="org/Mid-1", model=object(), tokenizer=None, device="cpu", letters={}
    )
    rungs.release(FakeModel("org/Small-1"))
    assert rungs.router is not None
