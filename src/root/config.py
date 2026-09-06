from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

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
    return local if local.is_file() else PACKAGED_CONFIGS / name


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    max_steps: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.3
    force_first_call: bool = False

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError(f"agent {self.name}: system_prompt is empty")
        if self.max_steps < 1:
            raise ValueError(f"agent {self.name}: max_steps must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError(f"agent {self.name}: max_new_tokens must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"agent {self.name}: temperature must be in [0, 2]")
        if self.force_first_call and not self.tools:
            raise ValueError(f"agent {self.name}: force_first_call needs at least one tool")


def load_agents(path: Path) -> dict[str, AgentSpec]:
    """Read configs/agents.yaml into validated specs.

    `defaults` applies to every agent; per-agent keys override it. Unknown keys
    are an error rather than a silently ignored typo.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = document.get("defaults") or {}
    raw_agents = document.get("agents") or {}
    if not raw_agents:
        raise ValueError(f"no agents defined in {path}")

    allowed = {f.name for f in fields(AgentSpec)} - {"name"}
    specs: dict[str, AgentSpec] = {}
    for name, overrides in raw_agents.items():
        merged = {**defaults, **(overrides or {})}
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
    return written
