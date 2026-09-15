"""Send a prompt to a model sized for it.

A ladder of models costs nothing to describe and a lot to choose between. The
choice is made here by a small model reading the prompt and answering one
multiple-choice question, scored off its logits in a single forward pass - no
generation, no parsing.

Three tiers are routed, not four, and that is measured rather than assumed.
Asked to sort prompts into trivial / ordinary / hard / research, Qwen3.5-0.8B
separates the first three and cannot separate the last two:

    trivial   0.70 - 0.96
    ordinary  1.06 - 1.32    gap +0.10
    hard      1.48 - 1.82    gap +0.16
    research  1.76 - 1.89    gap -0.06, overlapping

So the top tier is never chosen for you. Reaching for a model that size is a
decision about money and latency that a 0.8B reading one sentence has not
earned; ask for it with /tier xlarge.

Over a wider set of 26 prompts the hard boundary stays clean and the cheap one
does not quite:

    trivial   0.72 - 1.24   (n=12)
    ordinary  1.05 - 1.29   (n=8)
    hard      1.55 - 1.87   (n=6)

22 of the 26 land on the right rung at the thresholds below, which is the most
any pair of thresholds gets. Every miss is a trivial prompt sent to medium:
nothing was routed below its label, so a miss costs some speed rather than an
answer the model was too small to give.

LFM2.5-350M was measured as a router too and separates nothing - every gap
negative, A scoring higher than D. The router has to be the ~1B class.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backend import LocalModel, pick_device
from .config import ModelChoice, models_document, resolve_model
from .engines import TRANSFORMERS, load_engine

logger = logging.getLogger(__name__)

# What /tier and --tier accept beside a tier name: leave the session on one
# model, or let the router choose per prompt.
OFF = "off"

# The rungs, cheapest first, sized for roughly <1B, <7B, <100B and beyond.
# `xlarge` is configurable and selectable but never predicted.
TIERS = ("small", "medium", "large", "xlarge")
ROUTED = TIERS[:3]

CHOICES = ("A", "B", "C", "D")
QUESTION = (
    "Rate how much reasoning power a request needs. Answer with one letter.\n"
    "A = trivial: arithmetic, a date, listing files, reformatting text.\n"
    "B = ordinary: explain a concept, write a short function, summarise.\n"
    "C = hard: design a system, refactor with tradeoffs, multi-step analysis.\n"
    "D = research: build a whole pipeline, novel algorithm work, long projects."
)

# Midpoints of the measured gaps above. Tuning them against the 26-prompt set
# lands on (1.00, 1.30), which scores the same, so they are left where the
# gaps put them.
THRESHOLDS = (1.01, 1.40)


@dataclass(frozen=True, slots=True)
class Rung:
    """One step of the ladder: which model answers, and where it runs.

    `engine` is empty when the rung runs wherever the session already is. The
    large rungs have to name one: a 27B model on the local engine would try to
    put tens of gigabytes of weights in this process.
    """

    tier: str
    model: str
    engine: str = ""
    url: str = ""


@dataclass(frozen=True, slots=True)
class Ladder:
    rungs: dict[str, Rung] = field(default_factory=dict)
    router: str = ""

    @property
    def routed(self) -> list[str]:
        """The tiers the router may pick, cheapest first."""
        return [tier for tier in ROUTED if tier in self.rungs]

    @property
    def usable(self) -> bool:
        """Whether routing can do anything - one rung has nothing to choose."""
        return bool(self.router) and len(self.routed) > 1


def load_ladder(document: dict) -> Ladder:
    """Read the `router:` and `tiers:` keys of models.yaml.

    A tier may be a bare alias or a mapping with `engine` and `url`. Tiers left
    out are left out rather than empty, so a two-rung ladder routes between two
    rungs instead of failing on a third that was never configured.
    """
    entries = document.get("tiers") or {}
    unknown = set(entries) - set(TIERS)
    if unknown:
        raise ValueError(f"unknown tiers {sorted(unknown)}; expected {list(TIERS)}")

    rungs = {}
    for tier, entry in entries.items():
        if not entry:
            continue
        if isinstance(entry, str):
            entry = {"model": entry}
        if not entry.get("model"):
            raise ValueError(f"tier {tier}: no model named")
        rungs[tier] = Rung(
            tier=tier,
            model=str(entry["model"]),
            engine=str(entry.get("engine") or ""),
            url=str(entry.get("url") or ""),
        )
    return Ladder(rungs=rungs, router=str(document.get("router") or ""))


def read_ladder(path: Path | None = None) -> Ladder:
    return load_ladder(models_document(path))


def score_to_tier(score: float, available: list[str]) -> str:
    """Which configured tier a router score falls in.

    The boundaries are taken in order against the rungs that exist, so a ladder
    missing its middle rung splits at the first boundary rather than leaving its
    top rung unreachable.
    """
    if not available:
        raise ValueError("no tiers configured")
    for boundary, tier in zip(THRESHOLDS, available[:-1], strict=False):
        if score < boundary:
            return tier
    return available[-1]


@dataclass(slots=True)
class Router:
    """The small model that reads a prompt and says how hard it looks.

    It reads logits rather than text, so it has to be local: a served engine
    returns completions, and one sampled letter throws away the distribution
    that makes the score continuous.
    """

    model_id: str
    model: object
    tokenizer: object
    device: str
    letters: dict[str, list[int]]
    owned: bool = True

    @classmethod
    def load(cls, model_id: str, device: str | None = None) -> Router:
        device = device or pick_device()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32 if device == "cpu" else torch.bfloat16
        ).to(device)
        model.eval()
        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            device=device,
            letters=_letter_ids(tokenizer),
        )

    @classmethod
    def wrap(cls, local: LocalModel) -> Router:
        """Route with a model the session already has in memory.

        The router and the small rung are usually the same weights, and a second
        copy of them is a gigabyte spent to answer the same question.
        """
        return cls(
            model_id=local.model_id,
            model=local.model,
            tokenizer=local.tokenizer,
            device=local.device,
            letters=_letter_ids(local.tokenizer),
            owned=False,
        )

    def score(self, prompt: str) -> float:
        """How hard the prompt looks, on the 0-3 scale the choices define.

        The expected value over the four letters rather than the argmax: the
        letters are ordered, so a prompt the router splits between B and C
        belongs between them, not in whichever won by a hair.
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": QUESTION}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]
        probabilities = logits.float().softmax(dim=-1)
        weights = [
            sum(probabilities[token].item() for token in self.letters[letter]) for letter in CHOICES
        ]
        total = sum(weights)
        if not total:
            raise ValueError(f"{self.model_id} put no mass on any of {list(CHOICES)}")
        return sum(index * weight for index, weight in enumerate(weights)) / total

    def unload(self) -> None:
        """Free the weights, unless they belong to a model still in use."""
        if not self.owned:
            return
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


@dataclass(slots=True)
class Rungs:
    """The models a ladder has actually brought up.

    Rungs are opened when a prompt first needs them and then kept: a session
    that alternates between two tiers should not pay to load either one twice.
    """

    ladder: Ladder
    models: dict[str, ModelChoice]
    device: str | None = None
    engine: str = TRANSFORMERS
    url: str = ""
    router: Router | None = None
    loaded: dict[str, object] = field(default_factory=dict)
    adopted: set[str] = field(default_factory=set)

    def adopt(self, model) -> None:
        """Claim an already-loaded model for any rung that names it.

        The session comes up on one model before routing is switched on, and
        that model is usually the small rung.
        """
        for tier, rung in self.ladder.rungs.items():
            if rung.engine and rung.engine != self.engine:
                continue
            if resolve_model(rung.model, self.models).id == getattr(model, "model_id", ""):
                if self.loaded.setdefault(tier, model) is model:
                    self.adopted.add(tier)

    def release(self, model) -> None:
        """Forget a model the session is about to unload.

        `/model` and `/inference` free the session's weights. A rung that
        adopted them would otherwise keep handing back a model whose tensors
        are gone.
        """
        for tier in [tier for tier in self.adopted if self.loaded.get(tier) is model]:
            del self.loaded[tier]
            self.adopted.discard(tier)
        if self.router is not None and not self.router.owned:
            if self.router.model is getattr(model, "model", None):
                self.router = None

    def open(self, tier: str):
        if tier not in self.ladder.rungs:
            have = [name for name in TIERS if name in self.ladder.rungs]
            raise KeyError(f"no {tier} tier configured; have {have}")
        if tier not in self.loaded:
            rung = self.ladder.rungs[tier]
            choice = resolve_model(rung.model, self.models)
            kwargs = {"trust_remote_code": choice.trust_remote_code}
            if rung.url or self.url:
                kwargs["base_url"] = rung.url or self.url
            self.loaded[tier] = load_engine(
                choice.id, rung.engine or self.engine, device=self.device, **kwargs
            )
        return self.loaded[tier]

    def ready(self) -> Router:
        """The router, loaded if it is not up yet.

        The router is usually one of the rungs as well. Opening that rung first
        and routing with its weights costs nothing extra; loading the router on
        its own would put a second copy of the same model on the device.
        """
        if self.router is not None:
            return self.router
        wanted = resolve_model(self.ladder.router, self.models).id
        shared = next(
            (
                model
                for model in self.loaded.values()
                if isinstance(model, LocalModel) and model.model_id == wanted
            ),
            None,
        )
        if shared is None:
            tier = self._local_rung(wanted)
            shared = self.open(tier) if tier else None
        self.router = (
            Router.wrap(shared)
            if isinstance(shared, LocalModel)
            else Router.load(wanted, self.device)
        )
        return self.router

    def _local_rung(self, model_id: str) -> str | None:
        """The tier that runs this model in this process, if one does."""
        for tier, rung in self.ladder.rungs.items():
            if (rung.engine or self.engine) != TRANSFORMERS:
                continue
            if resolve_model(rung.model, self.models).id == model_id:
                return tier
        return None

    def pick(self, prompt: str) -> tuple[object, str, float]:
        """The model for this prompt, the tier it came from, and the score."""
        score = self.ready().score(prompt)
        tier = score_to_tier(score, self.ladder.routed)
        return self.open(tier), tier, score

    def unload(self) -> None:
        if self.router is not None:
            self.router.unload()
            self.router = None
        for tier, model in self.loaded.items():
            # An adopted rung is the session's own model; freeing it here would
            # leave the session without one.
            if tier not in self.adopted:
                model.unload()
        self.loaded.clear()
        self.adopted.clear()


def _letter_ids(tokenizer) -> dict[str, list[int]]:
    """Token ids for each choice letter, bare and space-prefixed.

    Which of the two a model emits first depends on its chat template, not on
    its answer, so both count towards the letter.
    """
    ids = {}
    for letter in CHOICES:
        found = {
            encoded[0]
            for encoded in (
                tokenizer.encode(form, add_special_tokens=False) for form in (letter, f" {letter}")
            )
            if len(encoded) == 1
        }
        if not found:
            raise ValueError(f"{letter!r} is not a single token for this router")
        ids[letter] = sorted(found)
    return ids
