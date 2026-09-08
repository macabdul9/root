from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, fields
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

# Used only when models.yaml is missing or names an alias it does not define.
FALLBACK_MODEL = "LiquidAI/LFM2.5-350M"

PACKAGED_CONFIGS = Path(__file__).parent / "defaults"
LOCAL_CONFIGS = Path("configs")


def config_path(name: str, override: Path | None = None) -> Path:
    """Where to read one config file from.

    An explicit path wins, then a `configs/` directory beside the working
    directory, then the copy shipped inside the package. That order is what lets
    `root` work from anywhere once installed, while a checkout or a project with
    its own `configs/` still overrides the defaults.
    """
    if override is not None:
        return override
    local = LOCAL_CONFIGS / name
    if not local.is_file():
        return PACKAGED_CONFIGS / name
    # Worth saying out loud. A local copy that has drifted from the packaged
    # one silently changed an eval result twice while it looked like a model
    # regression, and nothing pointed at the file being read.
    logger.warning("using %s, which shadows the packaged %s", local, name)
    return local


@dataclass(frozen=True, slots=True)
class Decoding:
    """How tokens are sampled.

    `top_k` and `repetition_penalty` are off at their neutral values rather than
    absent, so a config that sets one reads the same as one that does not.
    """

    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 256
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be 0 (off) or positive")
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError("min_p must be in [0, 1)")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

    @property
    def samples(self) -> bool:
        return self.temperature > 0

    def as_generate_kwargs(self) -> dict[str, object]:
        """The subset transformers should see.

        Greedy decoding ignores the sampling knobs, and passing them anyway
        makes transformers warn on every call.
        """
        kwargs: dict[str, object] = {"max_new_tokens": self.max_new_tokens}
        if self.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if not self.samples:
            kwargs["do_sample"] = False
            return kwargs
        kwargs |= {"do_sample": True, "temperature": self.temperature, "top_p": self.top_p}
        if self.top_k:
            kwargs["top_k"] = self.top_k
        if self.min_p:
            kwargs["min_p"] = self.min_p
        return kwargs


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    max_steps: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    force_first_call: bool = False
    preamble: str = ""

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError(f"agent {self.name}: system_prompt is empty")
        if self.max_steps < 1:
            raise ValueError(f"agent {self.name}: max_steps must be >= 1")
        if self.force_first_call and not self.tools:
            raise ValueError(f"agent {self.name}: force_first_call needs at least one tool")
        try:
            # Building it validates every decoding knob, with the agent named.
            _ = self.decoding
        except ValueError as exc:
            raise ValueError(f"agent {self.name}: {exc}") from None

    @property
    def decoding(self) -> Decoding:
        return Decoding(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
        )


def load_agents(path: Path) -> dict[str, AgentSpec]:
    """Read configs/agents.yaml into validated specs.

    `defaults` applies to every agent and `preamble` is prepended to every
    agent's prompt; per-agent keys override the defaults. Unknown keys are an
    error rather than a silently ignored typo.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = document.get("defaults") or {}
    raw_agents = document.get("agents") or {}
    if not raw_agents:
        raise ValueError(f"no agents defined in {path}")

    preamble = (document.get("preamble") or "").strip()
    allowed = {f.name for f in fields(AgentSpec)} - {"name"}
    specs: dict[str, AgentSpec] = {}
    for name, overrides in raw_agents.items():
        merged = {"preamble": preamble, **defaults, **(overrides or {})}
        unknown = set(merged) - allowed
        if unknown:
            raise ValueError(f"agent {name}: unknown keys {sorted(unknown)}")
        merged["tools"] = tuple(merged.get("tools") or ())
        specs[name] = AgentSpec(name=name, **merged)
    return specs


@dataclass(frozen=True, slots=True)
class ModelChoice:
    alias: str
    id: str
    note: str = ""
    trust_remote_code: bool = False


def _read_models_config(path: Path) -> dict:
    """Missing file is not an error: --model still takes any Hugging Face id."""
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_models(path: Path | None = None) -> dict[str, ModelChoice]:
    """Read the offered models, keyed by short alias."""
    document = _read_models_config(config_path("models.yaml", path))
    return {
        alias: ModelChoice(
            alias=alias,
            id=entry["id"],
            note=entry.get("note", ""),
            trust_remote_code=bool(entry.get("trust_remote_code", False)),
        )
        for alias, entry in (document.get("models") or {}).items()
    }


def resolve_model(name: str, models: dict[str, ModelChoice]) -> ModelChoice:
    """Turn an alias into a model choice, passing unknown names through as ids."""
    known = models.get(name)
    if known:
        return known
    listed = next((choice for choice in models.values() if choice.id == name), None)
    return listed or ModelChoice(alias=name, id=name)


def load_default_model(path: Path | None = None) -> str:
    """The model id named by `default:` in models.yaml.

    The config file is the one place the default lives, so changing which model
    starts is an edit there rather than in two files that can disagree.
    """
    resolved = config_path("models.yaml", path)
    document = _read_models_config(resolved)
    chosen = load_models(resolved).get(document.get("default", ""))
    return chosen.id if chosen else FALLBACK_MODEL


def copy_defaults(destination: Path = LOCAL_CONFIGS) -> list[Path]:
    """Write the packaged configs into a directory so they can be edited.

    The eval fixture comes too. Copying only the YAML leaves evals.yaml
    pointing at a workspace that is not there, and every file-reading case then
    fails against an empty directory rather than saying anything is wrong.

    Existing files are left alone: this is for starting from the defaults, not
    for discarding local changes.
    """
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for source in sorted(PACKAGED_CONFIGS.glob("*.yaml")):
        target = destination / source.name
        if target.exists():
            continue
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(target)

    fixture = PACKAGED_CONFIGS / "workspace"
    if fixture.is_dir() and not (destination / "workspace").exists():
        shutil.copytree(fixture, destination / "workspace")
        written.append(destination / "workspace")
    return written


HUB_API = "https://huggingface.co/api/models"
HUB_TIMEOUT = 15
# root is for models that fit on a laptop beside everything else you are
# running. Anything larger belongs behind a served engine, not in-process.
MAX_PARAMETERS = 1_000_000_000


def alias_for(model_id: str) -> str:
    """A short handle for a Hugging Face id.

    `LiquidAI/LFM2.5-350M` becomes `lfm2.5-350m`: the repository name, lowered,
    with anything that would be awkward in a config key flattened to a dash.
    """
    name = model_id.rstrip("/").split("/")[-1].lower()
    handle = re.sub(r"-+", "-", re.sub(r"[^a-z0-9.]+", "-", name)).strip("-.")
    # The tuning suffix is not what distinguishes one entry from another, and
    # `qwen2.5-coder-0.5b-instruct` is a lot to type at a prompt.
    for suffix in ("-instruct", "-chat", "-it"):
        handle = handle.removesuffix(suffix)
    return handle


def hub_parameters(model_id: str) -> int | None:
    """Parameter count from the hub, or None when it does not say."""
    try:
        response = requests.get(f"{HUB_API}/{model_id}", timeout=HUB_TIMEOUT)
        if response.status_code >= 400:
            return None
        return (response.json().get("safetensors") or {}).get("total")
    except requests.RequestException:
        return None


def check_size(model_id: str, parameters: int | None = None) -> None:
    """Refuse a model too large to run in process.

    A count the hub does not publish is allowed through rather than guessed at:
    the alternative is refusing every model whose metadata is incomplete.
    """
    total = parameters if parameters is not None else hub_parameters(model_id)
    if total is not None and total > MAX_PARAMETERS:
        raise ValueError(
            f"{model_id} has {total / 1e9:.2f}B parameters, over the {MAX_PARAMETERS / 1e9:.0f}B "
            "limit root runs in process. Serve it with vllm or ollama and use --engine instead."
        )


def describe_hub_model(model_id: str) -> str:
    """Look a model up on the hub and summarise it for the config.

    Also the existence check: a typo in the id is far easier to fix here than
    after a failed download. A note of "" means the hub had nothing to add.
    """
    response = requests.get(f"{HUB_API}/{model_id}", timeout=HUB_TIMEOUT)
    if response.status_code in {401, 403, 404}:
        # An unauthenticated lookup cannot tell a typo from a private repo:
        # both come back 401, so the message has to cover both.
        raise ValueError(
            f"Hugging Face has no readable model at {model_id!r}. "
            "Check the spelling, or set HF_TOKEN if it is private or gated."
        )
    response.raise_for_status()
    body = response.json()

    parts = []
    parameters = (body.get("safetensors") or {}).get("total")
    check_size(model_id, parameters)
    if parameters:
        parts.append(f"{parameters / 1e9:.2f}B params".replace("0.", "0."))
    task = body.get("pipeline_tag")
    if task and task != "text-generation":
        parts.append(task)
    if body.get("gated"):
        parts.append("gated, needs a token")
    return ", ".join(parts)


def register_model(
    model_id: str,
    alias: str | None = None,
    note: str = "",
    trust_remote_code: bool = False,
    path: Path | None = None,
) -> ModelChoice:
    """Add a model to the local models.yaml, creating it from the packaged one.

    Registration always writes to `./configs`, never into the installed
    package: a `configs/` directory beside the working directory is what
    `config_path` already prefers, so a registered model wins over the defaults.
    """
    target = path or LOCAL_CONFIGS / "models.yaml"
    if not target.is_file():
        copy_defaults(target.parent)

    handle = alias or alias_for(model_id)
    existing = load_models(target)
    if handle in existing:
        raise ValueError(f"{handle!r} is already registered as {existing[handle].id}")

    entry = {"id": model_id}
    if note:
        entry["note"] = note
    if trust_remote_code:
        entry["trust_remote_code"] = True

    _append_model(target, handle, entry)
    return ModelChoice(alias=handle, id=model_id, note=note, trust_remote_code=trust_remote_code)


def unregister_model(alias: str, path: Path | None = None) -> str:
    """Drop a model from the local models.yaml."""
    target = path or LOCAL_CONFIGS / "models.yaml"
    document = _read_models_config(target)
    models = document.get("models") or {}
    if alias not in models:
        raise ValueError(f"{alias!r} is not registered in {target}")
    removed = models.pop(alias)["id"]
    if document.get("default") == alias:
        document["default"] = next(iter(models), "")
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return removed


def _append_model(target: Path, alias: str, entry: dict) -> None:
    """Add one entry, keeping the comments in the file where possible.

    Appending text preserves them; re-dumping the parsed document does not. The
    append is verified by reading the file back, and falls back to a dump if the
    file is not shaped the way that assumes.
    """
    original = target.read_text(encoding="utf-8")
    lines = [f"  {alias}:", f"    id: {entry['id']}"]
    if entry.get("note"):
        lines.append(f"    note: {entry['note']}")
    if entry.get("trust_remote_code"):
        lines.append("    trust_remote_code: true")
    appended = original.rstrip("\n") + "\n" + "\n".join(lines) + "\n"

    target.write_text(appended, encoding="utf-8")
    if alias in load_models(target):
        return

    target.write_text(original, encoding="utf-8")
    document = _read_models_config(target)
    document.setdefault("models", {})[alias] = entry
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
